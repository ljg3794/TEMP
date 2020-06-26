import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.callbacks as callbacks
from data_util import *
from model import CNNClassifier
from utils.args import model_args,data_args
import pickle,random,re

physical_devices = tf.config.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu,True)


with open("vocab.pkl","rb") as f:
    vocab = pickle.load(f)

for key,val in data_args.items():
    globals().update({key:val})

for key,val in model_args.items():
    globals().update({key:val})

vocab_size = len(vocab)

config = {
    "vocab_size":vocab_size,
    "winsize":5,
    "emb_dim":100,
    "num_labels":2,
}

all_dom_lst = read_lines(os.path.join(data_txt_path, all_dom_file))
test_dom_lst= read_lines(os.path.join(data_txt_path ,test_dom_file))
train_dom_lst = sorted(list(set(all_dom_lst)-set(test_dom_lst)))
    
fname_gen=file_creator(train_dom_lst,"train")
fname_gen_dev=file_creator(train_dom_lst,"dev")
fname_gen_test=file_creator(train_dom_lst,"test")

self_score = dict()
self_history = dict()

start=time.time()
for _fname,_fname_dev,_fname_test in zip(fname_gen,fname_gen_dev,fname_gen_test):
    fname = os.path.join(data_id_path,_fname)
    fname_dev = os.path.join(data_id_path,_fname_dev)
    fname_test = os.path.join(data_id_path,_fname_test)
    
    if not os.path.exists(csv_path):
        os.mkdir(csv_path)
    
    if not os.path.exists(ckpt_path):
        os.mkdir(os.path.join(ckpt_path))
        if not os.path.exists(os.path.join(ckpt_path,_fname)):
            os.mkdir(os.path.join(ckpt_path,_fname))

    ckpt_name  =  os.path.join(ckpt_path,_fname,"cp.ckpt")
    csv_name  =  os.path.join(csv_path,_fname)
        
    csv_callback = callbacks.CSVLogger(csv_name)
    ckpt_callback = callbacks.ModelCheckpoint(ckpt_name,save_best_only=True)
    
    dataset = cls_dataset(fname,128,True,1000)
    eval_dataset = cls_dataset(fname_dev,128,True,1000)
    test_dataset = cls_dataset(fname_test,128,True,1000)
    
    print("\nTask : {}\n".format(_fname))
    model = CNNClassifier(config,200)
    loss=tf.keras.losses.sparse_categorical_crossentropy
    model.compile(loss=loss,metrics="accuracy")
    history=model.fit(dataset,epochs=10,validation_data= eval_dataset
        ,verbose=0,callbacks=[csv_callback,ckpt_callback])
    test_result = model.evaluate(test_dataset,verbose=1)
    self_score[_fname] = test_result[1]
    self_history[_fname] = history
end=time.time()
print("Classifier training : {:.4f} secs took".format(end-start))

    

def split_task_usage(string):
    task,star,usage = re.split(r"\.",string)
    task = ".".join([task,star])
    return task,usage

def fine_tune_and_score(base,target,config):
    print("Transfer : {} -> {}".format(base,target))
    target_task,_ = split_task_usage(target)
    target_test_task = target_task+".test"
    
    base_name = os.path.join(data_id_path,base)
    target_name = os.path.join(data_id_path,target)
    target_test_name = os.path.join(data_id_path,target_test_task)

    dataset = cls_dataset(target_name,64,True,1000)
    dataset_test = cls_dataset(target_test_name,64,True,1000)

    model_finetune = CNNClassifier(config,200)
    model_finetune.load_weights(os.path.join(ckpt_path,base,"cp.ckpt"))
    loss=tf.keras.losses.sparse_categorical_crossentropy
    model_finetune.compile(loss=loss,metrics="accuracy")

    for layer in model_finetune.layers:
        layer.trainable=False
    layer.trainable=True
    history = model_finetune.fit(dataset,epochs=3,validation_data=dataset_test)
    return history



# select a few of tasks and calculate transfer task
task_list = sorted(os.listdir(ckpt_path))
n=m=len(task_list)
selected_entries = random.sample([(i,j) for i in range(n) for j in range(m) if i!=j], int(n*m*0.6))
score_matrix = np.zeros(shape=[n,m]).astype(np.float32)

start=time.time()
for i,task_i in enumerate(task_list):
    print("{} -> others".format(task_i))
    for j,task_j in enumerate(task_list):
        if i==j:
            pass
        elif (i,j) in selected_entries :
            history = fine_tune_and_score(task_i,task_j,config)
            score = np.max(history.history['val_accuracy'])
            score_matrix[i,j]=score

for i in range(n):
    score_matrix[i,i] = self_score[task_list[i]]

selected_entries = selected_entries+[(i,i) for i in range(n)]
end=time.time()
print("Transfer matrix training : {:.4f} secs took".format(end-start))



# matrix 전처리(scoring matrix, make unreliable entries etc)
def column_wise_stat(mat):
    mean_lst = []
    std_lst = []
    n,m = mat.shape
    for col in range(m):
        col_vec=mat[:,col]
        mean_lst.append(np.mean(col_vec))
        std_lst.append(np.std(col_vec))
    return mean_lst, std_lst

n,m = score_matrix.shape
means,stds =column_wise_stat(score_matrix) 
p1,p2= 0.5, 0.5

Y = np.zeros_like(score_matrix).astype(np.float32)
reliable_idx = []
for i,j in selected_entries:
    if score_matrix[i,j] > means[j] + p1*stds[j] and score_matrix[j,i] > means[i] + p1*stds[i]:
        Y[i,j]=1.0
        Y[j,i]=1.0
        reliable_idx.append((i,j))
        reliable_idx.append((j,i))
    elif score_matrix[i,j] < means[j] - p2*stds[j] and score_matrix[j,i] < means[i] - p2*stds[i]:
        Y[i,j]=.0
        Y[j,i]=.0
        reliable_idx.append((i,j))
        reliable_idx.append((j,i))   
    else:
        pass

print("{}/{} is observed, and {}/{} is reliable".format(np.sum(score_matrix!=0.),n*m,len(reliable_idx),np.sum(score_matrix!=0.)))

# Matrix completion
import cvxpy as cp
import numpy as np

def noise_mat_gen(n,m,scailer=10):
    return np.random.randn(n,m)/scailer

def ei(i,length):
    out = np.zeros([length]).astype(np.float32)
    out[i] = 1.
    return out

n,m = Y.shape
sym_score = np.maximum(Y,Y.transpose() )

lamb = cp.Parameter(nonneg=True)
X = cp.Variable(Y.shape,symmetric=True,value = sym_score)
E = cp.Variable(Y.shape)#,value = noise_mat_gen(n,m))

observed_idx = np.argwhere(Y!=0) # idx (i,j), where reliable entries are
constraints = [ei(i,n)@(X+E)@ei(j,m)==Y[i,j] for i,j in observed_idx]

lamb.value = 1e2
obj = cp.Minimize(cp.norm(X,"nuc") + lamb*cp.norm(E,1))

prob = cp.Problem(obj,constraints)
optimal_value = prob.solve()

# Cluster
num_cluster=5
import sklearn.cluster as cluster
Cluster =  cluster.SpectralClustering(num_cluster)
cluster_idx = Cluster.fit_predict(X.value)

for i in range(num_cluster):
    print("\nCluster {} :".format(i))
    print(" / ".join(np.array(task_list)[cluster_idx==i]))
    

# Save the cluster as txt file(or json)
json_dict = dict()
for cluster in range(num_cluster):
    cluster_list = list(np.array(task_list)[cluster_idx==cluster])
    cluster_list = list(map(split_task_usage,cluster_list))
    json_dict[str(cluster)] = cluster_list

with open("cluster.json","w") as f:
    json.dump(json_dict,f,indent=4)


