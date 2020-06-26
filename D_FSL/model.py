import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
"""
bs=4 ; vocab_size=10 ; emb_dim=5 ; h_dim=10;winsize=5;seq_len=10

embedded = tf.random.normal([bs,seq_len,embed_dim])
conv_layer = layers.Conv1D(h_dim,winsize)
encoder = CNNEncoder(embed_dim,h_dim,winsize)
encoder(tf.random.normal([bs,seq_len,emb_dim]))
"""
"""
config = {
    "vocab_size":
    "winsize":5,
    "emb_dim":100
}
"""


class CNNEncoder(layers.Layer):
    
    def __init__(self,config,filters,padding="valid",activation="relu",is_proj=False,is_resid=False):
        super(CNNEncoder,self).__init__()
        self.config = config
        self.is_resid=is_resid
        self.conv1d_layer = layers.Conv1D(filters,self.config["winsize"],padding=padding,activation="relu")
    
    def call(self,inp):
        x = self.conv1d_layer(inp)
        if self.is_resid:
            x = x+inp
        return x

class CNNClassifier(keras.Model):
    
    def __init__(self,config,filters,padding="valid",activation="relu",is_proj=False,proj_dim=None):
        super(CNNClassifier,self).__init__()
        self.config = config
        self.embedding = layers.Embedding(self.config["vocab_size"], self.config["emb_dim"])
        self.is_proj = is_proj
        if self.is_proj :
            self.proj_fc = layers.Dense(proj_dim)
        self.conv1d_layer = layers.Conv1D(filters,self.config["winsize"],activation=activation
            ,padding=padding)
        self.fc = layers.Dense(self.config["num_labels"],activation="softmax")

    def call(self,inp):
        x = self.embedding(inp)
        if self.is_proj:
            x = self.proj_fc(x)
        x = self.conv1d_layer(x)
        x = tf.reduce_max(x,1)
        x = self.fc(x)
        return x
        
    def num_flat_features(self, x):
        size = x.shape[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



def cosine_similarity(x, y, dim=1,minval=1e-6):
    w12 = tf.matmul(x,y,transpose_b=True)
    w1 = tf.norm(x,axis=1,keepdims=True)
    w2 = tf.reshape(tf.norm(y,axis=1),[1,-1])
    pairwise = w12 / tf.matmul(w1,w2)
    return tf.clip_by_value(pairwise,1e-6,1e+08)

class MatchingLayer(layers.Layer):
    
    def __init__(self, nonlinear='softmax', use_cosine=False):
        super(MatchingLayer, self).__init__()
        self.use_cosine = use_cosine
        try:
            self.activation = getattr(keras.activations,nonlinear)
        except:
            self.activation = None
        
    def call(self,inp):
        if self.use_cosine:
            sim = cosine_similarity(inp.x, inp.y, dim=1)
        else:
            sim = tf.matmul(inp.x,inp.y,transpose_b=True)
        if self.activation is not None:
            output = self.activation(sim)
        else:
            output = sim
        return output
    

class MatchPair():
    def __init__(self,x,y):
        self.x = x
        self.y = y

class MatchingCnn(keras.Model):
    
    def __init__(self,config,filters,is_resid,padding="valid",use_cosine=True,additional_proj=False,filters2=None):

        super(MatchingCnn,self).__init__()
        self.config = config
        self.additional_proj = additional_proj
        if is_resid:
            padding="SAME"
        else:
            padding="valid"


        self.column_embed = layers.Embedding(config["vocab_size"],config["emb_dim"])
        self.column_encoder = keras.Sequential()
        self.column_encoder.add(CNNEncoder(config, filters,is_resid=is_resid,padding=padding))
        if additional_proj:
            self.column_encoder.add(CNNEncoder(config,filters2,is_resid=is_resid,padding=padding))
        self.match_classifier = MatchingLayer(nonlinear="softmax",use_cosine=use_cosine)
    
    def call(self,inp,is_avg=False,n_way=2,n_support=5):
        emb = self.column_embed(inp.x)
        hidden = self.column_encoder(emb)
        hidden = tf.reduce_max(hidden,1)

        emb2 = self.column_embed(inp.y)
        hidden2 = self.column_encoder(emb2)
        hidden2 = tf.reduce_max(hidden2,1)

        if is_avg:
            hidden = tf.reduce_mean(tf.reshape(hidden,[n_way,n_support,-1]),axis=1)
        
        hiddens = MatchPair(hidden,hidden2)
        
        output = self.match_classifier(hiddens)

        return output

    def get_hidden(self,x):
        emb = self.column_embed(x)
        hidden = self.column_encoder(emb)
        return hidden.numpy()

    def num_flat_features(self, x):
        size = x.shape[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    
    
    
    
    
    
    
    
