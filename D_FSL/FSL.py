import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import random,os,pickle,json
#from utils.args import model_args,data_args
from model import MatchingCnn,MatchPair
from data_util import * 
from utils.args import *

physical_devices = tf.config.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu,True)

with open("vocab.pkl","rb") as f:
    vocab = pickle.load(f)

vocab_size = len(vocab)
config = {
    "vocab_size":vocab_size,
    "winsize":5,
    "emb_dim":100,
    "num_labels":2,
}

with open("cluster.json","r") as f:
    cluster_dict = json.load(f)

n_query=15;n_way=2;k_shot=5 ; num_epi=1000 ; num_epochs=15
config_fsl={
    "n_query":15,
    "n_way":2,
    "k_shot":5,
    "num_epi":1000,
    "is_avg":True
}

config_fsl.update(model_args)
config_fsl.update({
    "vocab_size":vocab_size,
    "winsize":5,
    "emb_dim":100,
    "num_labels":2
})

fsl_ckpt_dir = "ckpt_log_fsl"
if not os.path.exists(fsl_ckpt_dir):
    os.mkdir(fsl_ckpt_dir)
alpha_dir = "alpha"
if not os.path.exists(alpha_dir):
    os.mkdir(alpha_dir)

def train_cluster_fsl(config,cluster,num_epochs,num_epi):
    n_query = config["n_query"]
    n_way = config["n_way"]
    k_shot = config["k_shot"]
    num_epi = config["num_epi"]
    
    optimizer = tf.keras.optimizers.Adam()
    mcnn = MatchingCnn(config,100,is_resid=True)
    loss = tf.keras.losses.sparse_categorical_crossentropy
    
    for epc in range(num_epochs):
        acc_metric = tf.keras.metrics.Mean()
        gen = get_batch_FSL2(cluster,"train",n_way,k_shot,n_query)
        start=time.time()
        for step,(support_set,query_set) in enumerate(gen):
            labels=label_gen(n_way,k_shot,n_query,config["is_avg"])
            inp=MatchPair(support_set,query_set)
            with tf.GradientTape() as tape:
                out = mcnn(inp,is_avg=config["is_avg"],n_way=n_way,n_support=k_shot)
                out = tf.transpose(out)
                loss_val = tf.reduce_mean(loss(labels,out))
                acc_metric(tf.argmax(out,axis=1)==labels)
            if step % 100 == 0:
                print("Epoch {}, Step {}, Loss : {}".format(epc, step, loss_val.numpy()))
            grads= tape.gradient(loss_val,mcnn.trainable_weights)
            optimizer.apply_gradients(zip(grads,mcnn.trainable_weights))
        print("Mean Acc :",acc_metric.result().numpy())
        end=time.time()
    print("Epoch {}, took {:.4f} secs".format(epc+1,end-start))
    return mcnn
    
cluster_fsl_lst =[]
for i,cluster in cluster_dict.items():
    cluster_fsl_lst.append(train_cluster_fsl(config_fsl,cluster,15,None))

for i,cluster in cluster_dict.items():
    
    ckpt_path = os.path.join(fsl_ckpt_dir,"cluster_"+str(i))
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)
    ckpt_name = os.path.join(ckpt_path,"ck.ckpt")
    cluster_fsl_lst[int(i)].save_weights(ckpt_name)

if len(cluster_fsl_lst) == 0:
    listdir = sorted(os.listdir(fsl_ckpt_dir))
    for ckpt_dir in listdir: 
        ckpt_dir = os.path.join(fsl_ckpt_dir,ckpt_dir)
        mcnn = MatchingCnn(config,100,is_resid=True)
        mcnn.load_weights(ckpt_dir)
        cluster_fsl_lst.append(mcnn)
         
# FSL on meta-testing data
test_dom_lst= read_lines(os.path.join("data_txt","workspace.target.list"))
_lst = []
for task in test_dom_lst:
    _lst.append(task+".t2")
    _lst.append(task+".t4")
    _lst.append(task+".t5")
test_dom_lst = _lst

task,usage,n_way,n_query,root_dir=root_data_dir,num_epi=None
def metaTest_fsl(target_task,num_epochs,n_way,n_query):
    mcnn_linear_comb = FSL_wrapper(num_cluster=5)
    for epc in range(num_epochs):
        gen = get_batch_FSL3(target_task,"dev",n_way,n_query)        
        loss_func = tf.keras.losses.~
        for support,query in gen:
            labels = label_gen(n_way,5,n_query,is_avg)
            inp = MatchPair(support,query)
            with tf.GradientTape() as tape: 
                out = mcnn_linear_comb(inp)
                loss_val = loss_func(labels,out)
            grads = tape.gradient(loss_val,mcnn_linear_comb.trainable_weights)
            optimizer.apply_gradients(zip(grads,mcnn_linear_comb.trainable_weights))
        
    return mcnn_linear_comb    

class FSL_wrapper(keras.Model):
    
    def __init__(self,model_list):
        """
        Note that model_list the instance of cluster MatchNet
        """
        super(FSL_wrapper,self).__init__()
        self.K = len(model_list)
        alpha_val = np.reshape(np.array(np.array([1.]*self.K)/self.K),[self.K,1,1])
        self.alpha =tf.Variable(alpha_val,dtype=tf.float32,trainable=True)

        for model in model_list:
            model.trainable=False
            
        self.cluster_models = model_list
    
    def call(self,inp,is_avg,n_way=2,k_shot=5):
        model_results = [tf.transpose(model(inp,is_avg,n_way,k_shot)) for model in self.cluster_models]
        model_results = tf.reshape(model_results,[self.K,-1,2])
        final_results = tf.reduce_sum(self.alpha*model_results,axis=0)
        return final_results

    def save_alpha(self,path):
        X=self.alpha.numpy()
        np.save(path,X)
    
    def load_alpha(self,path):
        _alpha = np.load(path).astype(np.float32)
        _alpha = tf.Tensor(_alpha,dtype=tf.float32)
        self.alpha.assign(_alpha)



num_epochs = 20
loss = tf.keras.losses.sparse_categorical_crossentropy
fsl_wrapper_lst = []
for target_task in test_dom_lst:
    print("\nTarget task :",target_task)
    current_alpha_dir= os.path.join(alpha_dir,target_task)
    if not os.path.exists(current_alpha_dir):
        os.mkdir(current_alpha_dir)
    fsl_wrap = FSL_wrapper(cluster_fsl_lst)
    optimizer = tf.keras.optimizers.Adam()
    for epc in range(num_epochs):
        acc_metric = tf.keras.metrics.Mean()
        gen = get_batch_FSL3(target_task,"dev",n_way,n_query) 
        loss = tf.keras.losses.sparse_categorical_crossentropy
        for support,query in gen:
            labels = label_gen(n_way,5,n_query,config_fsl["is_avg"])
            inp = MatchPair(support,query)
            with tf.GradientTape() as tape:
                out = fsl_wrap(inp,True,n_way,k_shot)
                loss_val = loss(labels,out)
            grads = tape.gradient(loss_val,fsl_wrap.trainable_weights)
            optimizer.apply_gradients(zip(grads,fsl_wrap.trainable_weights))
            acc_metric(tf.argmax(out,axis=1)==labels)
        fsl_wrap.save_alpha(current_alpha_dir+"/alpha.npy")
        print("Mean Acc :",acc_metric.result().numpy())

    fsl_wrapper_lst.append(fsl_wrap)

    
for target,target_fsl in zip(test_dom_lst,fsl_wrapper_lst):
    gen_test = get_batch_FSL3(target_task,"test",n_way,n_query,num_epi=1000) 
    acc_metric = tf.keras.metrics.Mean()
    for support,query in gen_test:
        inp = MatchPair(support,query)
        out = target_fsl(inp,True,n_way,k_shot)
        acc_metric(tf.argmax(out,axis=1)==labels)
    print("\n(Target,Test)",target,"Mean Acc :",acc_metric.result().numpy(),"\n")

for target_task in test_dom_lst:
    current_alpha_dir= os.path.join(alpha_dir,target_task)
    alpha =np.squeeze(np.load(current_alpha_dir+"/alpha.npy"))
    print("{} : {}".format(target_task,list(alpha)))
