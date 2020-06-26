# 결국 데이터는
# 1. 클래스별로 파일을 저장
# 2. 임의로 선택된 한 클래스에서, num_support + n_query만큼 (랜덤하게) 불러옴(각각, support와 query을 형성하는데 사용됌)
# 3. 2번작업을 n_way만큼 반복하여 최종 episode을 형성
# 4. label은, 해당 episode에서의 label이므로, 전체 class의 depth인 one-hot 벡터일 필요는 없다(즉, n_way 깊이의 원핫벡터이 n_way * n_query만큼 존재)
# 5. loss에 대해서는, support set(shape=[num_classes,embed_dim])에 대해서 모든 query(num_classes*n_query개)의 pairwise similarity을 구함, 즉  [num_classes*nquery , num_classes]의 매트릭스형성(각 query에 대한 logit이라고 볼 수 있다 computed by a support set). 이제 label이 n_way*n_query이므로, cross entropy loss구하기 가능

# 따라서, class별로 텍스트 id을 구하고 있어야할 것으로 보임(즉, id-converted txt 가 필요)


# util functions
import random,os,time
import tensorflow as tf
import numpy as np
from utils.args import data_args
from collections import Counter,defaultdict
from math import ceil

root_data_dir="data_id"

for key,val in data_args.items():
    globals().update({key:val})

def count_rows(path):
    cnt=0
    f = open(path)
    for i in f:
        cnt+=1
    f.close()
    return cnt

def read_and_parse(txt_path,dtype=str,is_shuffle=False):
    txt_lst = []
    label_lst = []
    f = open(txt_path,"r")
    for line in f:
        line = line.strip()
        txt,label = parse_line(line,dtype)
        label_lst.append(label)
        txt_lst.append(txt)
    if is_shuffle:
        num= len(label_lst)
        assert num == len(txt_lst)
        idx = random.sample(random.sample(range(num),num),num)
        txt_lst = [txt_lst[i] for i in idx]
        label_lst = [label_lst[i] for i in idx]
    f.close()
    return txt_lst,label_lst

def parse_line(line,dtype=int):
    label = dtype(line[-2:])
    ids = list(map(lambda w:dtype(w),line[:-2].split()))
    return ids,label

def selected_read(path,idx):
    """
    Read only idx lines from a file in path
    """
    id_lst = [] ; label_lst = []
    idx_len =len(idx)
    cnt = 0
    f = open(path,"r")
    for i,line in enumerate(f):
        if i in idx:
            line = line.strip()
            ids,label=parse_line(line)
            id_lst.append(ids)
            label_lst.append(label)
            cnt += 1
        if cnt >= idx_len:
            break
    f.close()
    return id_lst,label_lst
    

def read_lines(path):
    """
    Only read lines(No parse etc, just strip and read)
    """
    f = open(path,"r")
    lines=[]
    for line in f:
        line = line.strip()
        lines.append(line)
    f.close()
    return lines

def count_update(txt_lst,counter):
    """
    txt_lst : a list of tokeinized sentences
    counter : vocabulary counter to be updated
    """
    for sent in txt_lst:
        counter.update(sent)


def file_creator(domain_lst,usage="train"):
    """
    domain_lst : a domain lst(apparel, book,...)
    usage : train,dev, or test
    """
    task_by_star = ["t2","t4","t5"]
    if type(usage) == str:
        usage = [usage]
    for domain in domain_lst:
        for t in task_by_star:
            for use in usage:
                task_star=".".join([domain,t,use])
                yield task_star

def suffix_to_list(lst,suffix,delim="."):
    return list(map(lambda x:x+delim+suffix,lst))

def prefix_to_list(lst,prefix,delim="."):
    return list(map(lambda x:prefix+delim+x,lst))

def to_ids(txt,vocab,dtype=int):
    return list(map(lambda w:dtype(vocab.get(w,vocab["<unk>"])),txt))

def to_sent(ids,rev_vocab):
    return " ".join(list(map(lambda w:rev_vocab.get(w,"<unk>"),ids)))

def pad_to_max(lst):
    max_len=0
    for elem in lst:
        if len(elem)>=max_len:
            max_len = len(elem)
    
    _lst = [elem+[0]*(max_len-len(elem)) for elem in lst]
    return _lst

def get_batch_CLS(txt_path,bs):
    txt_lst,label_lst = read_and_parse(txt_path,dtype=int,is_shuffle=True)
    total_num = len(label_lst)
    num_iter = 1+(total_num//bs)
    for i in range(1,num_iter+1):
        if i % num_iter == 0:
            yield txt_lst[(i-1)*bs:],label_lst[(i-1)*bs:]
        else:
            yield txt_lst[(i-1)*bs:i*bs],label_lst[(i-1)*bs:i*bs]

def get_batch_CLS_for_tf(txt_path):
    txt_lst,label_lst = read_and_parse(txt_path,dtype=int,is_shuffle=True)
    total_num = len(label_lst)
    for i in range(0,total_num):
        yield txt_lst[i],label_lst[i]


def cls_dataset(path,batch_size,is_shuffle,buffer_size=1000):
    data= tf.data.Dataset.from_generator(get_batch_CLS_for_tf,
        output_types=(tf.int32,tf.int32),
        output_shapes=([None,],[]),args=(path,))
    if is_shuffle:
        data_padded = data.shuffle(buffer_size).padded_batch(32)
    else:
        data_padded = data.padded_batch(32)
    return data_padded

def get_batch_FSL2(task_list,usage,n_way,k_shot,n_query,root_dir=root_data_dir,num_epi=None):
    task_list = suffix_to_list(task_list,usage)
    task_file_list = prefix_to_list(task_list,root_dir,"/")
    
    total_txt=[];total_label=[]
    
    for i in range(len(task_file_list)):
        txt,label = read_and_parse(task_file_list[i],int,is_shuffle=False)
        total_txt.extend(txt)
        total_label.extend(label)
    
    total_txt = np.array(total_txt)
    total_label = np.array(total_label)
    if num_epi is None:
        total_cases = len(total_txt)
        num_epi = ceil(total_cases/(n_way*n_query))
        print("Total {} sentences, {} episodes".format(total_cases,num_epi))
    else:
        print("Total {} sentences, {} episodes".format(total_cases,num_epi))
    zero_idx = np.argwhere(total_label==0)[:,0]
    one_idx = np.argwhere(total_label==1)[:,0]
    
    for i in range(num_epi):
        selected_zeros = np.random.choice(zero_idx,k_shot+n_query)
        selected_zeros_support = selected_zeros[:k_shot]
        selected_zeros_query = selected_zeros[k_shot:]

        selected_ones = np.random.choice(one_idx,k_shot+n_query)
        selected_ones_support = selected_ones[:k_shot]
        selected_ones_query = selected_ones[k_shot:]
    
        support_set= np.array(pad_to_max(
            np.hstack([total_txt[selected_zeros_support],total_txt[selected_ones_support]])
         ))

        query_set = np.array(pad_to_max(
            np.hstack([total_txt[selected_zeros_query],total_txt[selected_ones_query]])
        ))
    
        labels=np.repeat(list(range(n_way)),n_query)
        
        yield support_set,query_set 

def get_batch_FSL3(task,usage,n_way,n_query,root_dir=root_data_dir,num_epi=None):
    """
    Support_set fixed version(For meta-testing)
    """
    query_file = [os.path.join(root_dir,task+"."+usage)]
    support_file = os.path.join(root_dir,task+".train") 
    
    query_txt=[];query_label=[]
    
    for i in range(len(query_file)):
        txt,label = read_and_parse(query_file[i],int,is_shuffle=False)
        query_txt.extend(txt)
        query_label.extend(label)

    query_txt = np.array(query_txt)
    query_label = np.array(query_label)
    
    support_set,support_label = read_and_parse(support_file,int)
    support_set = np.array(pad_to_max(support_set))
    support_label = np.array(support_label)
    
    if num_epi is None:
        total_cases = len(query_txt)
        num_epi = ceil(total_cases/(n_way*n_query))
        print("Total {} sentences, {} episodes".format(total_cases,num_epi))
    
    zero_idx = np.argwhere(query_label==0)[:,0]
    one_idx = np.argwhere(query_label==1)[:,0]
    
    for i in range(num_epi):
        selected_zeros = np.random.choice(zero_idx,n_query)

        selected_ones = np.random.choice(one_idx,n_query)
    
        query_set = np.array(pad_to_max(
            np.hstack([query_txt[selected_ones],query_txt[selected_zeros]])
        ))
    
        labels=np.repeat(list(range(n_way)),n_query)
        
        yield support_set,query_set 

def label_gen(nw,ks,nq,is_avg):
    if is_avg:
        labels =np.repeat(np.arange(nw),nq)
        labels = np.reshape(labels,[nq*nw,])
    else:
        labels = np.repeat(np.arange(nw),ks*nq)
        labels = np.reshape(labels,[nq*nw,ks])
    return labels
"""
def get_batch_FSL(task_lst,num_epi,usage,n_way,k_shot,n_query):
    task_lst = suffix_to_list(task_lst,usage)
    max_row_lst = []
    
    for task in task_lst:
        f_path = os.path.join(data_id_path,task)
        max_row_lst.append(count_rows(f_path))
    
    for epi in range(num_epi):
        task_selected_idx = random.sample(range(len(task_lst)),n_way)
        support = []
        query = []
        #batch_label = np.tile(np.arange(n_way)[:,np.newaxis],(1,n_query)).astype(np.int32)
     
        for task_idx in task_selected_idx:
            f_path = os.path.join(data_id_path,task_lst[task_idx])
            n_row = max_row_lst[task_idx]
            selected_idx = random.sample(range(n_row),k_shot+n_query)
            txt_lst,_ = selected_read(f_path,selected_idx)
            support.extend(txt_lst[:k_shot])
            query.extend(txt_lst[k_shot:])
        
        support = pad_to_max(support)
        query= pad_to_max(query)
        yield support,query#,batch_label
"""
"""tf.data.Dataset API로 만들고 싶음
def get_batch_FSL(task_lst,usage,n_way,k_shot,n_query):
    task_lst =concat_usage(task_lst,usage)
    _dict={}
    for task in task_lst:
        txt_path = os.path.join(data_id_path,task)
        txt_lst,_ = read_and_parse(txt_path,dtype=int,is_shuffle=True)
        _dict.update({task:txt_lst})
    
    while True:
        selected_task = random.sample(list(_dict.keys()),n_way)
        for task in selected_task:
            selected_txt_lst = random.sample(txt_lst,k_shot+n_query)
            yield selected_txt_lst[:k_shot], selected_txt_lst[k_shot:]
"""


if __name__ == "__main__":
    """
    Build vocab, and turn (engilsh) text file into a tokenized ids text file, according to the
    vocabulary. For example, "This phone works fine"->"6 100 20 87"
    """
    from utils.args import data_args
    from collections import Counter
    import pickle

    for key,val in data_args.items():
        globals().update({key:val})

    all_dom_lst = read_lines(os.path.join(data_txt_path, all_dom_file))
    test_dom_lst= read_lines(os.path.join(data_txt_path ,test_dom_file))
    train_dom_lst = sorted(list(set(all_dom_lst)-set(test_dom_lst)))
    
    vocab_counter = Counter()
    
    gen = file_creator(train_dom_lst,"train") # Vocab is build only using the training data
    
    print("Build Vocabulary....")
    for txt_file in gen:
        txt_lst,label_lst = read_and_parse(os.path.join(data_txt_path,txt_file))
        count_update(txt_lst,vocab_counter)
    vocab_counter=vocab_counter.most_common()

    vocab = dict()
    vocab["<pad>"]=0 ; vocab["<unk>"]=1 ; vocab["<eos>"]=2
    for word,freq in vocab_counter:
        if freq >= min_freq:
            vocab[word]=len(vocab)
    print("Total Vocab Length  : {} -> {}".format(len(vocab_counter),len(vocab)))

    with open("vocab.pkl","wb") as f:
        pickle.dump(vocab,f)
    
    if not os.path.exists(data_id_path):
        os.mkdir(data_id_path)
    
    # Meta-training dataset
    print("(Training) Turn text data into ids.... in {}".format(data_id_path))
    gen=file_creator(train_dom_lst,"train")
    gen_dev=file_creator(train_dom_lst,"dev")
    gen_test=file_creator(train_dom_lst,"test")
    
    for txt_file in gen:
        f = open(os.path.join(data_id_path,txt_file),"w")
        txt_lst,label_lst = read_and_parse(os.path.join(data_txt_path ,txt_file) )
        for sent,label in zip(txt_lst,label_lst) : 
            if label == "-1": # "-1 for label" is trouble-causing for tensorflow
                 label = "0"
            sent_id = to_ids(sent,vocab,str)
            sent_id = " ".join(sent_id+[str(label)])+" \n"
            f.write(sent_id)
        f.close()
    f.close()
    
    for txt_file in gen_dev:
        f = open(os.path.join(data_id_path,txt_file),"w")
        txt_lst,label_lst = read_and_parse(os.path.join(data_txt_path ,txt_file) )
        for sent,label in zip(txt_lst,label_lst) : 
            if label == "-1":
                label = "0"
            sent_id = to_ids(sent,vocab,str)
            sent_id = " ".join(sent_id+[str(label)])+" \n"
            f.write(sent_id)
        f.close()
    f.close()

    for txt_file in gen_test:
        f = open(os.path.join(data_id_path,txt_file),"w")
        txt_lst,label_lst = read_and_parse(os.path.join(data_txt_path ,txt_file) )
        for sent,label in zip(txt_lst,label_lst) : 
            if label == "-1":
                label = "0"
            sent_id = to_ids(sent,vocab,str)
            sent_id = " ".join(sent_id+[str(label)])+" \n"
            f.write(sent_id)
        f.close()
    f.close()

    # Meta-testing dataset
    print("(Test) Turn text data into ids.... in {}".format(data_id_path))
    gen=file_creator(test_dom_lst,"train")
    gen_dev=file_creator(test_dom_lst,"dev")
    gen_test=file_creator(test_dom_lst,"test")
    
    for txt_file in gen:
        f = open(os.path.join(data_id_path,txt_file),"w")
        txt_lst,label_lst = read_and_parse(os.path.join(data_txt_path ,txt_file) )
        for sent,label in zip(txt_lst,label_lst) : 
            if label == "-1":
                label = "0"
            sent_id = to_ids(sent,vocab,str)
            sent_id = " ".join(sent_id+[str(label)])+" \n"
            f.write(sent_id)
        f.close()
    f.close()
    
    for txt_file in gen_dev:
        f = open(os.path.join(data_id_path,txt_file),"w")
        txt_lst,label_lst = read_and_parse(os.path.join(data_txt_path ,txt_file) )
        for sent,label in zip(txt_lst,label_lst) : 
            if label == "-1":
                label = "0"
            sent_id = to_ids(sent,vocab,str)
            sent_id = " ".join(sent_id+[str(label)])+" \n"
            f.write(sent_id)
        f.close()
    f.close()

    for txt_file in gen_test:
        f = open(os.path.join(data_id_path,txt_file),"w")
        txt_lst,label_lst = read_and_parse(os.path.join(data_txt_path ,txt_file) )
        for sent,label in zip(txt_lst,label_lst) : 
            if label == "-1":
                label = "0"
            sent_id = to_ids(sent,vocab,str)
            sent_id = " ".join(sent_id+[str(label)])+" \n"
            f.write(sent_id)
        f.close()
    f.close()
