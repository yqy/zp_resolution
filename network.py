#coding=utf8
import sys
import gzip
import cPickle

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample

from theano.compile.nanguardmode import NanGuardMode


import lasagne

#theano.config.exception_verbosity="high"
#theano.config.optimizer="fast_compile"

#aaaaa

'''
Deep neural network for AZP resolution
a kind of Memory Network
Created by qyyin 2016.11.10
'''

#activation function
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

#### Constants
GPU = True
if GPU:
    print >> sys.stderr,"Trying to run under a GPU. If this is not desired,then modify NetWork.py\n to set the GPU flag to False."
    try: theano.config.device = 'gpu'
    except: pass # it's already set 
    theano.config.floatX = 'float32'
else:
    print >> sys.stderr,"Running with a CPU. If this is not desired,then modify the \n NetWork.py to set\nthe GPU flag to True."
    theano.config.floatX = 'float64'

def init_weight(n_in,n_out,activation_fn=sigmoid,pre="",uni=True,ones=False):
    rng = np.random.RandomState(1234)
    if uni:
        W_values = np.asarray(rng.normal(size=(n_in, n_out), scale= .01, loc = .0), dtype = theano.config.floatX)
    else:
        W_values = np.asarray(
            rng.uniform(
                low=-np.sqrt(6. / np.sqrt(n_in + n_out)),
                high=np.sqrt(6. / np.sqrt(n_in + n_out)),
                size=(n_in, n_out)
            ),
            dtype=theano.config.floatX
        )
        if activation_fn == theano.tensor.nnet.sigmoid:
            W_values *= 4
            W_values /= 6

    b_values = np.zeros((n_out,), dtype=theano.config.floatX)

    if ones:
        b_values = np.ones((n_out,), dtype=theano.config.floatX)

    w = theano.shared(
        value=W_values,
        name='%sw'%pre, borrow=True
    )
    b = theano.shared(
        value=b_values,
        name='%sb'%pre, borrow=True
    )
    return w,b

class Layer():
    def __init__(self,n_in,n_out,inpt,activation_fn=tanh):
        self.params = []
        if inpt:
            self.inpt = inpt
        else:
            self.inpt= T.matrix("inpt")
        self.w,self.b = init_weight(n_in,n_out,pre="MLP_")
        self.params.append(self.w) 
        self.params.append(self.b) 
    
        self.output = activation_fn(T.dot(self.inpt, self.w) + self.b)

class LinearLayer():
    def __init__(self,n_in,n_out,inpt,activation_fn=tanh):
        self.params = []
        if inpt:
            self.inpt = inpt
        else:
            self.inpt= T.matrix("inpt")
        self.w,self.b = init_weight(n_in,n_out,pre="MLP_")
        self.params.append(self.w) 
        #self.params.append(self.b) 
    
        self.output = T.dot(self.inpt, self.w)

def _dropout_from_layer(layer, p=0.5):
    """p is the probablity of dropping a unit
    """
    rng = np.random.RandomState(1234)
    srng = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output

class NetWork():
    def __init__(self,n_hidden,embedding_dimention=50,hope_num=1):

        ##n_in: sequence lstm 的输入维度
        ##n_hidden: lstm for candi and zp 的隐层维度
        ##n_hidden_sequence: sequence lstm的隐层维度 因为要同zp的结合做dot，所以其维度要是n_hidden的2倍
        ##                   即 n_hidden_sequence = 2 * n_hidden
        self.params = []

        #self.dot_output_dropout = _dropout_from_layer(self.dot_output,p=0.2)

        self.zp_x_pre = T.matrix("zp_x_pre")
        self.zp_x_post = T.matrix("zp_x_post")
        
        #self.zp_x_pre_dropout = _dropout_from_layer(self.zp_x_pre)
        #self.zp_x_post_dropout = _dropout_from_layer(self.zp_x_post)


        zp_nn_pre = LSTM(embedding_dimention,n_hidden,self.zp_x_pre)
        #zp_nn_pre = LSTM(embedding_dimention,n_hidden,self.zp_x_pre_dropout)
        self.params += zp_nn_pre.params
        
        zp_nn_post = LSTM(embedding_dimention,n_hidden,self.zp_x_post)
        #zp_nn_post = LSTM(embedding_dimention,n_hidden,self.zp_x_post_dropout)
        self.params += zp_nn_post.params

        self.zp_out = T.concatenate((zp_nn_pre.nn_out,zp_nn_post.nn_out))
        self.zp_out_dropout = _dropout_from_layer(T.concatenate((zp_nn_pre.nn_out,zp_nn_post.nn_out)))
        
        self.get_zp_out = theano.function(inputs=[self.zp_x_pre,self.zp_x_post],outputs=[self.zp_out])
    
        ### get sequence output for NP ###
        self.np_x = T.tensor3("np_x")
        #self.np_x_dropout = _dropout_from_layer(self.np_x)

        self.mask = T.matrix("mask")
    
        self.np_nn_in = LSTM_batch(embedding_dimention,n_hidden*2,self.np_x,self.mask)
        #self.np_nn_in = LSTM_batch(embedding_dimention,n_hidden*2,self.np_x_dropout,self.mask)
        self.params += self.np_nn_in.params

        self.np_nn_out = LSTM_batch(embedding_dimention,n_hidden*2,self.np_x,self.mask)
        #self.np_nn_out = LSTM_batch(embedding_dimention,n_hidden*2,self.np_x_dropout,self.mask)
        self.params += self.np_nn_out.params


        #self.np_out = self.np_nn.nn_out
        self.np_in_output = (self.np_nn_in.all_hidden).mean(axis=1)
        self.np_in_output_dropout = _dropout_from_layer((self.np_nn_in.all_hidden).mean(axis=1))
        #self.np_out_output = (self.np_nn_out.all_hidden).mean(axis=1)
        self.np_out_output = self.np_nn_out.nn_out
        self.get_np_out = theano.function(inputs=[self.np_x,self.mask],outputs=[self.np_in_output])

        
        ### get attention for ZP and NP ###

        #new_zp = self.zp_out
        
        #nz,_ = theano.scan(self.attention_step,outputs_info=new_zp,non_sequences=[self.np_out],n_steps=hope_num)
        
        #dot = self.np_out*nz[-1]

        #dot_0 = self.zp_out*self.np_out
        dot_0 = self.zp_out*self.np_in_output
        dot_0_dropout = self.zp_out_dropout*self.np_in_output_dropout

        attention_0 = softmax(T.sum(dot_0,axis=[1]))[0] 
        attention_0_dropout = softmax(T.sum(dot_0_dropout,axis=[1]))[0] 

        zp_0 = T.sum(attention_0[:,None]*self.np_out_output,axis=0) + self.zp_out
        zp_0_dropout = T.sum(attention_0_dropout[:,None]*self.np_out_output,axis=0) + self.zp_out


        dot = zp_0*self.np_in_output
        dot_dropout = zp_0_dropout*self.np_in_output_dropout

        #self.get_dot = theano.function(inputs=[self.zp_x_pre,self.zp_x_post,self.np_x,self.mask],outputs=[T.sum(dot,axis=[1])])
        
        attention = softmax(T.sum(dot,axis=[1]))[0] 
        attention_dropout = softmax(T.sum(dot_dropout,axis=[1]))[0] 
    
        new_zp = T.sum(attention[:,None]*self.np_out_output,axis=0) + self.zp_out


        self.get_dot = theano.function(inputs=[self.zp_x_pre,self.zp_x_post,self.np_x,self.mask],outputs=[T.sum(dot,axis=[1])])
        self.get_attention = theano.function(inputs=[self.zp_x_pre,self.zp_x_post,self.np_x,self.mask],outputs=[attention])


        self.out = attention
        self.out_dropout = attention_dropout

        self.get_out = theano.function(inputs=[self.zp_x_pre,self.zp_x_post,self.np_x,self.mask],outputs=[self.out])

        
        l1_norm_squared = sum([(w**2).sum() for w in self.params])
        l2_norm_squared = sum([(abs(w)).sum() for w in self.params])

        lmbda_l1 = 0.0
        lmbda_l2 = 0.001
        #lmbda_l2 = 0.0

        t = T.bvector()
        #cost = -(T.log((self.out*t).sum()))
        cost = -(T.log((self.out_dropout*t).sum()))
        #cost = 1-((self.out*t).sum())

        self.get_cost = theano.function(inputs=[self.zp_x_pre,self.zp_x_post,self.np_x,self.mask,t],outputs=[cost])

        lr = T.scalar()
        #grads = T.grad(cost, self.params)
        #updates = [(param, param-lr*grad)
        #    for param, grad in zip(self.params, grads)]
        
        #updates = lasagne.updates.sgd(cost, self.params, lr)
        updates = lasagne.updates.adadelta(cost, self.params)

        
        self.train_step = theano.function(inputs=[self.zp_x_pre,self.zp_x_post,self.np_x,self.mask,t,lr], outputs=[cost],
            on_unused_input='warn',
            updates=updates)
            #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
            #) 

    def show_para(self):
        for para in self.params:
            print >> sys.stderr, para,para.get_value() 
    def attention_step(self,zp_y,np_y):
        #dot = self.np_out*self.zp_out 
        dot = zp_y*np_y
        #self.get_dot = theano.function(inputs=[self.zp_x_pre,self.zp_x_post,self.np_x,self.mask],outputs=[T.sum(dot,axis=[1])])
        
        attention = softmax(T.sum(dot,axis=[1]))[0] 
        
        #self.get_attention = theano.function(inputs=[self.zp_x_pre,self.zp_x_post,self.np_x,self.mask],outputs=[attention])
    
        new_zp = T.sum(attention[:,None]*np_y,axis=0)

        return new_zp


class NetWork_feature():
    def __init__(self,n_hidden,embedding_dimention=50,feature_dimention=61):

        ##n_in: sequence lstm 的输入维度
        ##n_hidden: lstm for candi and zp 的隐层维度
        ##n_hidden_sequence: sequence lstm的隐层维度 因为要同zp的结合做dot，所以其维度要是n_hidden的2倍
        ##                   即 n_hidden_sequence = 2 * n_hidden

        repre_active = ReLU

        self.params = []

        self.zp_x_pre = T.matrix("zp_x_pre")
        self.zp_x_post = T.matrix("zp_x_post")
        
        #self.zp_x_pre_dropout = _dropout_from_layer(self.zp_x_pre)
        #self.zp_x_post_dropout = _dropout_from_layer(self.zp_x_post)

        zp_nn_pre = GRU(embedding_dimention,n_hidden,self.zp_x_pre)
        #zp_nn_pre = LSTM(embedding_dimention,n_hidden,self.zp_x_pre_dropout)
        self.params += zp_nn_pre.params
        
        zp_nn_post = GRU(embedding_dimention,n_hidden,self.zp_x_post)
        #zp_nn_post = LSTM(embedding_dimention,n_hidden,self.zp_x_post_dropout)
        self.params += zp_nn_post.params

        self.zp_out = T.concatenate((zp_nn_pre.nn_out,zp_nn_post.nn_out))

        self.ZP_layer = Layer(n_hidden*2,n_hidden*2,self.zp_out,repre_active) 

        self.zp_out_output = self.ZP_layer.output

        #self.zp_out_dropout = _dropout_from_layer(T.concatenate((zp_nn_pre.nn_out,zp_nn_post.nn_out)))
        
        self.get_zp_out = theano.function(inputs=[self.zp_x_pre,self.zp_x_post],outputs=[self.ZP_layer.output])


        ### get sequence output for NP ###
        self.np_x = T.tensor3("np_x")
        self.np_x_post = T.tensor3("np_x")
        self.np_x_pre = T.tensor3("np_x")

        #self.np_x_dropout = _dropout_from_layer(self.np_x)

        self.mask = T.matrix("mask")
        self.mask_pre = T.matrix("mask")
        self.mask_post = T.matrix("mask")
    
        #self.np_nn_x = RNN_batch(embedding_dimention,n_hidden,self.np_x,self.mask)
        self.np_nn_x = GRU_batch(embedding_dimention,n_hidden,self.np_x,self.mask)
        self.params += self.np_nn_x.params
        self.np_nn_pre = GRU_batch(embedding_dimention,n_hidden,self.np_x_pre,self.mask_pre)
        self.params += self.np_nn_pre.params
        self.np_nn_post = GRU_batch(embedding_dimention,n_hidden,self.np_x_post,self.mask_post)
        self.params += self.np_nn_post.params

        self.np_nn_out = GRU_batch(embedding_dimention,n_hidden,self.np_x,self.mask)
        self.params += self.np_nn_out.params

        #self.np_nn_out = LSTM_batch(embedding_dimention,n_hidden*2,self.np_x,self.mask)
        #self.np_nn_out = LSTM_batch(embedding_dimention,n_hidden*2,self.np_x_dropout,self.mask)
        #self.params += self.np_nn_out.params


        #self.np_nn_x_output = (self.np_nn_x.all_hidden).mean(axis=1)
        self.np_nn_x_output = self.np_nn_x.nn_out
        self.np_nn_post_output = self.np_nn_post.nn_out
        self.np_nn_pre_output = self.np_nn_pre.nn_out

        self.np_out = T.concatenate((self.np_nn_x_output,self.np_nn_post_output,self.np_nn_pre_output),axis=1)

        self.NP_layer = Layer(n_hidden*3,n_hidden*2,self.np_out,repre_active) 

        self.np_out_output = self.NP_layer.output

        #self.np_x_head = T.transpose(self.np_x,axes=(1,0,2))[-1]
        self.np_x_head = (self.np_nn_out.all_hidden).mean(axis=1)

        self.get_np_head = theano.function(inputs=[self.np_x],outputs=[self.np_x_head])
        self.get_np = theano.function(inputs=[self.np_x,self.np_x_pre,self.np_x_post,self.mask,self.mask_pre,self.mask_post],outputs=[self.np_out])
        self.get_np_out = theano.function(inputs=[self.np_x,self.np_x_pre,self.np_x_post,self.mask,self.mask_pre,self.mask_post],outputs=[self.np_out_output])

        self.feature = T.matrix("feature")
        self.feature_layer = Layer(feature_dimention,n_hidden,self.feature,repre_active) 

        w_attention_zp,b_attention = init_weight(n_hidden*2,1,pre="attention_hidden",ones=False) 
        self.params += [w_attention_zp,b_attention]

        w_attention_np,b_u = init_weight(n_hidden*2,1,pre="attention_zp",ones=False) 
        self.params += [w_attention_np]

        w_attention_feature,b_u = init_weight(n_hidden,1,pre="attention_feature",ones=False) 
        self.params += [w_attention_feature]

        self.calcu_attention = tanh(T.dot(self.np_out_output,w_attention_np) + T.dot(self.zp_out_output,w_attention_zp) + T.dot(self.feature_layer.output,w_attention_feature) + b_attention)

        self.attention = softmax(T.transpose(self.calcu_attention,axes=(1,0)))[0]
        self.get_attention = theano.function(inputs=[self.zp_x_pre,self.zp_x_post,self.np_x,self.np_x_pre,self.np_x_post,self.mask,self.mask_pre,self.mask_post,self.feature],outputs=[self.attention])

        new_zp = T.sum(self.attention[:,None]*self.np_x_head,axis=0)
        self.get_new_zp = theano.function(inputs=[self.zp_x_pre,self.zp_x_post,self.np_x,self.np_x_pre,self.np_x_post,self.mask,self.mask_pre,self.mask_post,self.feature],outputs=[new_zp])

        #### *** HOP *** ####
        self.w_hop_zp,self.b_hop_zp = init_weight(n_hidden*2+embedding_dimention,n_hidden*2,pre="hop_")
        self.params += [self.w_hop_zp,self.b_hop_zp]


        ## hop 1 ##                
        self.zp_hop_1_init = T.concatenate((zp_nn_pre.nn_out,zp_nn_post.nn_out,new_zp))
        self.zp_hop_1 = repre_active(T.dot(self.zp_hop_1_init, self.w_hop_zp) + self.b_hop_zp)

        self.calcu_attention_hop_1 = tanh(T.dot(self.np_out_output,w_attention_np) + T.dot(self.zp_hop_1,w_attention_zp) + T.dot(self.feature_layer.output,w_attention_feature) + b_attention)
        self.attention_hop_1 = softmax(T.transpose(self.calcu_attention_hop_1,axes=(1,0)))[0]
        self.get_attention_hop_1 = theano.function(inputs=[self.zp_x_pre,self.zp_x_post,self.np_x,self.np_x_pre,self.np_x_post,self.mask,self.mask_pre,self.mask_post,self.feature],outputs=[self.attention_hop_1])


        self.out = self.attention_hop_1

        self.get_out = theano.function(inputs=[self.zp_x_pre,self.zp_x_post,self.np_x,self.np_x_pre,self.np_x_post,self.mask,self.mask_pre,self.mask_post,self.feature],outputs=[self.out])

        
        l1_norm_squared = sum([(w**2).sum() for w in self.params])
        l2_norm_squared = sum([(abs(w)).sum() for w in self.params])

        lmbda_l1 = 0.0
        #lmbda_l2 = 0.001
        lmbda_l2 = 0.0

        t = T.bvector()
        cost = -(T.log((self.out*t).sum()))
        #cost = -(T.log((self.out_dropout*t).sum()))
        #cost = 1-((self.out*t).sum())

        lr = T.scalar()
        #grads = T.grad(cost, self.params)
        #updates = [(param, param-lr*grad)
        #    for param, grad in zip(self.params, grads)]
        
        #updates = lasagne.updates.sgd(cost, self.params, lr)
        updates = lasagne.updates.adadelta(cost, self.params)

        
        self.train_step = theano.function(
            inputs=[self.zp_x_pre,self.zp_x_post,self.np_x,self.np_x_pre,self.np_x_post,self.mask,self.mask_pre,self.mask_post,self.feature,t,lr],
            outputs=[cost],
            on_unused_input='warn',
            updates=updates)
            #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
            #) 

    def show_para(self):
        for para in self.params:
            print >> sys.stderr, para,para.get_value() 



class NetWork_new():
    def __init__(self,n_hidden,embedding_dimention=50):

        ##n_in: sequence lstm 的输入维度
        ##n_hidden: lstm for candi and zp 的隐层维度
        ##n_hidden_sequence: sequence lstm的隐层维度 因为要同zp的结合做dot，所以其维度要是n_hidden的2倍
        ##                   即 n_hidden_sequence = 2 * n_hidden
        self.params = []

        self.zp_x_pre = T.matrix("zp_x_pre")
        self.zp_x_post = T.matrix("zp_x_post")
        
        #self.zp_x_pre_dropout = _dropout_from_layer(self.zp_x_pre)
        #self.zp_x_post_dropout = _dropout_from_layer(self.zp_x_post)

        zp_nn_pre = GRU(embedding_dimention,n_hidden,self.zp_x_pre)
        #zp_nn_pre = LSTM(embedding_dimention,n_hidden,self.zp_x_pre_dropout)
        self.params += zp_nn_pre.params
        
        zp_nn_post = GRU(embedding_dimention,n_hidden,self.zp_x_post)
        #zp_nn_post = LSTM(embedding_dimention,n_hidden,self.zp_x_post_dropout)
        self.params += zp_nn_post.params

        self.zp_out = T.concatenate((zp_nn_pre.nn_out,zp_nn_post.nn_out))

        self.ZP_layer = Layer(n_hidden*2,n_hidden*2,self.zp_out,ReLU) 

        self.zp_out_output = self.ZP_layer.output

        #self.zp_out_dropout = _dropout_from_layer(T.concatenate((zp_nn_pre.nn_out,zp_nn_post.nn_out)))
        
        self.get_zp_out = theano.function(inputs=[self.zp_x_pre,self.zp_x_post],outputs=[self.ZP_layer.output])


        ### get sequence output for NP ###
        self.np_x = T.tensor3("np_x")
        self.np_x_post = T.tensor3("np_x")
        self.np_x_pre = T.tensor3("np_x")

        #self.np_x_dropout = _dropout_from_layer(self.np_x)

        self.mask = T.matrix("mask")
        self.mask_pre = T.matrix("mask")
        self.mask_post = T.matrix("mask")
    
        self.np_nn_x = RNN_batch(embedding_dimention,n_hidden,self.np_x,self.mask)
        self.params += self.np_nn_x.params
        self.np_nn_pre = GRU_batch(embedding_dimention,n_hidden,self.np_x_pre,self.mask_pre)
        self.params += self.np_nn_pre.params
        self.np_nn_post = GRU_batch(embedding_dimention,n_hidden,self.np_x_post,self.mask_post)
        self.params += self.np_nn_post.params

        #self.np_nn_out = LSTM_batch(embedding_dimention,n_hidden*2,self.np_x,self.mask)
        #self.np_nn_out = LSTM_batch(embedding_dimention,n_hidden*2,self.np_x_dropout,self.mask)
        #self.params += self.np_nn_out.params


        #self.np_out = self.np_nn.nn_out
        self.np_nn_x_output = (self.np_nn_x.all_hidden).mean(axis=1)
        self.np_nn_post_output = self.np_nn_post.nn_out
        self.np_nn_pre_output = self.np_nn_pre.nn_out

        self.np_out = T.concatenate((self.np_nn_x_output,self.np_nn_post_output,self.np_nn_pre_output),axis=1)

        self.NP_layer = Layer(n_hidden*3,n_hidden*2,self.np_out,ReLU) 

        self.np_out_output = self.NP_layer.output

        self.np_x_head = T.transpose(self.np_x,axes=(1,0,2))[-1]

        self.get_np_head = theano.function(inputs=[self.np_x],outputs=[self.np_x_head])
        self.get_np = theano.function(inputs=[self.np_x,self.np_x_pre,self.np_x_post,self.mask,self.mask_pre,self.mask_post],outputs=[self.np_out])
        self.get_np_out = theano.function(inputs=[self.np_x,self.np_x_pre,self.np_x_post,self.mask,self.mask_pre,self.mask_post],outputs=[self.np_out_output])

        w_attention_zp,b_attention = init_weight(n_hidden*2,1,pre="attention_hidden",ones=False) 
        self.params += [w_attention_zp,b_attention]

        w_attention_np,b_u = init_weight(n_hidden*2,1,pre="attention_zp",ones=False) 
        self.params += [w_attention_np]

        self.calcu_attention = tanh(T.dot(self.np_out_output,w_attention_np) + T.dot(self.zp_out_output,w_attention_zp) + b_attention)
        self.attention = softmax(T.transpose(self.calcu_attention,axes=(1,0)))[0]
        self.get_attention = theano.function(inputs=[self.zp_x_pre,self.zp_x_post,self.np_x,self.np_x_pre,self.np_x_post,self.mask,self.mask_pre,self.mask_post],outputs=[self.attention])

        new_zp = T.sum(self.attention[:,None]*self.np_x_head,axis=0)
        self.get_new_zp = theano.function(inputs=[self.zp_x_pre,self.zp_x_post,self.np_x,self.np_x_pre,self.np_x_post,self.mask,self.mask_pre,self.mask_post],outputs=[new_zp])

        #### *** HOP *** ####
        self.w_hop_zp,self.b_hop_zp = init_weight(n_hidden*2+embedding_dimention,n_hidden*2,pre="hop_")
        self.params += [self.w_hop_zp,self.b_hop_zp]


        ## hop 1 ##                
        self.zp_hop_1_init = T.concatenate((zp_nn_pre.nn_out,zp_nn_post.nn_out,new_zp))
        self.zp_hop_1 = ReLU(T.dot(self.zp_hop_1_init, self.w_hop_zp) + self.b_hop_zp)

        self.calcu_attention_hop_1 = tanh(T.dot(self.np_out_output,w_attention_np) + T.dot(self.zp_hop_1,w_attention_zp) + b_attention)
        self.attention_hop_1 = softmax(T.transpose(self.calcu_attention_hop_1,axes=(1,0)))[0]
        self.get_attention_hop_1 = theano.function(inputs=[self.zp_x_pre,self.zp_x_post,self.np_x,self.np_x_pre,self.np_x_post,self.mask,self.mask_pre,self.mask_post],outputs=[self.attention_hop_1])


        self.out = self.attention_hop_1

        self.get_out = theano.function(inputs=[self.zp_x_pre,self.zp_x_post,self.np_x,self.np_x_pre,self.np_x_post,self.mask,self.mask_pre,self.mask_post],outputs=[self.out])

        
        l1_norm_squared = sum([(w**2).sum() for w in self.params])
        l2_norm_squared = sum([(abs(w)).sum() for w in self.params])

        lmbda_l1 = 0.0
        #lmbda_l2 = 0.001
        lmbda_l2 = 0.0

        t = T.bvector()
        cost = -(T.log((self.out*t).sum()))
        #cost = -(T.log((self.out_dropout*t).sum()))
        #cost = 1-((self.out*t).sum())

        lr = T.scalar()
        #grads = T.grad(cost, self.params)
        #updates = [(param, param-lr*grad)
        #    for param, grad in zip(self.params, grads)]
        
        #updates = lasagne.updates.sgd(cost, self.params, lr)
        updates = lasagne.updates.adadelta(cost, self.params)

        
        self.train_step = theano.function(
            inputs=[self.zp_x_pre,self.zp_x_post,self.np_x,self.np_x_pre,self.np_x_post,self.mask,self.mask_pre,self.mask_post,t,lr],
            outputs=[cost],
            on_unused_input='warn',
            updates=updates)
            #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
            #) 

    def show_para(self):
        for para in self.params:
            print >> sys.stderr, para,para.get_value() 


class LSTM_attention():
    def __init__(self,n_in,n_hidden,x=None,prefix=""):
         
        self.params = []
        if x:
            self.x = x
        else:
            self.x = T.matrix("x")

        wf_x,bf = init_weight(n_in,n_hidden,pre="%s_lstm_f_x_"%prefix,ones=True) 
        self.params += [wf_x,bf]

        wi_x,bi = init_weight(n_in,n_hidden,pre="%s_lstm_i_x_"%prefix) 
        self.params += [wi_x,bi]

        wc_x,bc = init_weight(n_in,n_hidden,pre="%s_lstm_c_x_"%prefix) 
        self.params += [wc_x,bc]

        wo_x,bo = init_weight(n_in,n_hidden,pre="%s_lstm_o_x_"%prefix) 
        self.params += [wo_x,bo]


        wf_h,b_h = init_weight(n_hidden,n_hidden,pre="%s_lstm_f_h_"%prefix)
        self.params += [wf_h]     

        wi_h,b_h = init_weight(n_hidden,n_hidden,pre="%s_lstm_i_h_"%prefix)
        self.params += [wi_h]     

        wc_h,b_h = init_weight(n_hidden,n_hidden,pre="%s_lstm_c_h_"%prefix)
        self.params += [wc_h]     

        wo_h,b_h = init_weight(n_hidden,n_hidden,pre="%s_lstm_o_h_"%prefix)
        self.params += [wo_h]     

        h_t_0 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX))
        c_t_0 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX))

        [h,c],r = theano.scan(self.lstm_recurrent_fn, sequences = self.x,
                       outputs_info = [h_t_0,c_t_0],
                       non_sequences = [wf_x,wf_h,bf,wi_x,wi_h,bi,wc_h,wc_x,bc,wo_x,wo_h,bo])

        #self.last_hidden = h[-1]
        self.all_hidden = h
        self.nn_out = h[-1]

    def lstm_recurrent_fn(self,x,h_t_1,c_t_1,wf_x,wf_h,bf,wi_x,wi_h,bi,wc_h,wc_x,bc,wo_x,wo_h,bo):
        ft = sigmoid(T.dot(h_t_1,wf_h) + T.dot(x,wf_x) + bf)

        it = sigmoid(T.dot(h_t_1,wi_h) + T.dot(x,wi_x) + bi)

        ot = sigmoid(T.dot(h_t_1,wo_h) + T.dot(x,wo_x) + bo)

        ct_ = tanh(T.dot(h_t_1,wc_h) + T.dot(x,wc_x) + bc)

        c_t = ft*c_t_1 + it*ct_

        h_t = ot*tanh(c_t)
        return h_t,c_t



class LSTM():
    def __init__(self,n_in,n_hidden,x=None,prefix=""):
         
        self.params = []
        self.x = x
        #if x:
        #    self.x = x
        #else:
        #    self.x = T.matrix("x")

        wf_x,bf = init_weight(n_in,n_hidden,pre="%s_lstm_f_x_"%prefix) 
        self.params += [wf_x,bf]

        wi_x,bi = init_weight(n_in,n_hidden,pre="%s_lstm_i_x_"%prefix) 
        self.params += [wi_x,bi]

        wc_x,bc = init_weight(n_in,n_hidden,pre="%s_lstm_c_x_"%prefix) 
        self.params += [wc_x,bc]

        wo_x,bo = init_weight(n_in,n_hidden,pre="%s_lstm_o_x_"%prefix) 
        self.params += [wo_x,bo]


        wf_h,b_h = init_weight(n_hidden,n_hidden,pre="%s_lstm_f_h_"%prefix)
        self.params += [wf_h]     

        wi_h,b_h = init_weight(n_hidden,n_hidden,pre="%s_lstm_i_h_"%prefix)
        self.params += [wi_h]     

        wc_h,b_h = init_weight(n_hidden,n_hidden,pre="%s_lstm_c_h_"%prefix)
        self.params += [wc_h]     

        wo_h,b_h = init_weight(n_hidden,n_hidden,pre="%s_lstm_o_h_"%prefix)
        self.params += [wo_h]     

        h_t_0 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX))
        c_t_0 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX))

        [h,c],r = theano.scan(self.lstm_recurrent_fn, sequences = self.x,
                       outputs_info = [h_t_0,c_t_0],
                       non_sequences = [wf_x,wf_h,bf,wi_x,wi_h,bi,wc_h,wc_x,bc,wo_x,wo_h,bo])

        #self.last_hidden = h[-1]
        self.all_hidden = h
        self.nn_out = h[-1]

    def lstm_recurrent_fn(self,x,h_t_1,c_t_1,wf_x,wf_h,bf,wi_x,wi_h,bi,wc_h,wc_x,bc,wo_x,wo_h,bo):
        ft = sigmoid(T.dot(h_t_1,wf_h) + T.dot(x,wf_x) + bf)

        it = sigmoid(T.dot(h_t_1,wi_h) + T.dot(x,wi_x) + bi)

        ot = sigmoid(T.dot(h_t_1,wo_h) + T.dot(x,wo_x) + bo)

        ct_ = tanh(T.dot(h_t_1,wc_h) + T.dot(x,wc_x) + bc)

        c_t = ft*c_t_1 + it*ct_

        h_t = ot*tanh(c_t)
        return h_t,c_t

class LSTM_batch():
    def __init__(self,n_in,n_hidden,x=T.tensor3("x"),mask=T.matrix("mask"),prefix=""):
         
        self.params = []
        if x is not None:
            self.x = x
        else:
            self.x = T.tensor3("x")

        if mask is not None:
            self.mask = mask
        else:
            self.mask = T.matrix("mask")

        #### 转置 为了进行scan运算 ###
    
        nmask = T.transpose(self.mask,axes=(1,0))
        nx = T.transpose(self.x,axes=(1,0,2))


        wf_x,bf = init_weight(n_in,n_hidden,pre="%s_lstm_f_x_"%prefix) 
        self.params += [wf_x,bf]

        wi_x,bi = init_weight(n_in,n_hidden,pre="%s_lstm_i_x_"%prefix) 
        self.params += [wi_x,bi]

        wc_x,bc = init_weight(n_in,n_hidden,pre="%s_lstm_c_x_"%prefix) 
        self.params += [wc_x,bc]

        wo_x,bo = init_weight(n_in,n_hidden,pre="%s_lstm_o_x_"%prefix) 
        self.params += [wo_x,bo]


        wf_h,b_h = init_weight(n_hidden,n_hidden,pre="%s_lstm_f_h_"%prefix)
        self.params += [wf_h]     

        wi_h,b_h = init_weight(n_hidden,n_hidden,pre="%s_lstm_i_h_"%prefix)
        self.params += [wi_h]     

        wc_h,b_h = init_weight(n_hidden,n_hidden,pre="%s_lstm_c_h_"%prefix)
        self.params += [wc_h]     

        wo_h,b_h = init_weight(n_hidden,n_hidden,pre="%s_lstm_o_h_"%prefix)
        self.params += [wo_h]     

        #h_t_0 = T.alloc(np.array(0.,dtype=np.float64), x.shape[0], n_hidden)
        #c_t_0 = T.alloc(np.array(0.,dtype=np.float64), x.shape[0], n_hidden)
        h_t_0 = T.alloc(0., x.shape[0], n_hidden)
        c_t_0 = T.alloc(0., x.shape[0], n_hidden)

        #h_t_0 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX))
        #c_t_0 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX))

        [h,c],r = theano.scan(self.lstm_recurrent_fn, sequences = [nx,nmask],
                       outputs_info = [h_t_0,c_t_0],
                       non_sequences = [wf_x,wf_h,bf,wi_x,wi_h,bi,wc_h,wc_x,bc,wo_x,wo_h,bo])

        self.all_hidden = T.transpose(h,axes=(1,0,2))
        self.nn_out = h[-1]

    def lstm_recurrent_fn(self,x,mask,h_t_1,c_t_1,wf_x,wf_h,bf,wi_x,wi_h,bi,wc_h,wc_x,bc,wo_x,wo_h,bo):
        ft = sigmoid(T.dot(h_t_1,wf_h) + T.dot(x,wf_x) + bf)

        it = sigmoid(T.dot(h_t_1,wi_h) + T.dot(x,wi_x) + bi)

        ot = sigmoid(T.dot(h_t_1,wo_h) + T.dot(x,wo_x) + bo)

        ct_ = tanh(T.dot(h_t_1,wc_h) + T.dot(x,wc_x) + bc)

        c_t_this = ft*c_t_1 + it*ct_

        h_t_this = ot*tanh(c_t_this)

        c_t = mask[:, None] * c_t_this + (1. - mask)[:, None] * c_t_1
        h_t = mask[:, None] * h_t_this + (1. - mask)[:, None] * h_t_1

        return h_t,c_t



class GRU():
    def __init__(self,n_in,n_hidden,x=None,prefix=""):
         
        self.params = []
        self.x = x

        wz_x,bz = init_weight(n_in,n_hidden,pre="%s_lstm_f_x_"%prefix) 
        self.params += [wz_x,bz]

        wr_x,br = init_weight(n_in,n_hidden,pre="%s_lstm_i_x_"%prefix) 
        self.params += [wr_x,br]

        wc_x,bc = init_weight(n_in,n_hidden,pre="%s_lstm_c_x_"%prefix) 
        self.params += [wc_x,bc]


        wz_h,b_h = init_weight(n_hidden,n_hidden,pre="%s_lstm_f_h_"%prefix)
        self.params += [wz_h]     

        wr_h,b_h = init_weight(n_hidden,n_hidden,pre="%s_lstm_i_h_"%prefix)
        self.params += [wr_h]     

        wc_h,b_h = init_weight(n_hidden,n_hidden,pre="%s_lstm_c_h_"%prefix)
        self.params += [wc_h]     


        h_t_1 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX))

        h,r = theano.scan(self.recurrent_fn, sequences = self.x,
                       outputs_info = [h_t_1],
                       non_sequences = [wz_x,wz_h,bz,wr_x,wr_h,br,wc_x,wc_h,bc])

        self.all_hidden = h
        self.nn_out = h[-1]

    def recurrent_fn(self,x,h_t_1,wz_x,wz_h,bz,wr_x,wr_h,br,wc_x,wc_h,bc):
        fz = sigmoid(T.dot(h_t_1,wz_h) + T.dot(x,wz_x) + bz)

        fr = sigmoid(T.dot(h_t_1,wr_h) + T.dot(x,wr_x) + br)

        h_new = tanh(T.dot(x,wc_x) + T.dot( (fr*h_t_1) ,wc_h) + bc)

        h_t = (1-fz)*h_t_1 + fz*h_new

        return h_t



class GRU_batch():
    def __init__(self,n_in,n_hidden,x=T.tensor3("x"),mask=T.matrix("mask"),prefix=""):
         
        self.params = []
        if x is not None:
            self.x = x
        else:
            self.x = T.tensor3("x")

        if mask is not None:
            self.mask = mask
        else:
            self.mask = T.matrix("mask")

        #### 转置 为了进行scan运算 ###
    
        nmask = T.transpose(self.mask,axes=(1,0))
        nx = T.transpose(self.x,axes=(1,0,2))


        wz_x,bz = init_weight(n_in,n_hidden,pre="%s_lstm_f_x_"%prefix) 
        self.params += [wz_x,bz]

        wr_x,br = init_weight(n_in,n_hidden,pre="%s_lstm_i_x_"%prefix) 
        self.params += [wr_x,br]

        wc_x,bc = init_weight(n_in,n_hidden,pre="%s_lstm_c_x_"%prefix) 
        self.params += [wc_x,bc]


        wz_h,b_h = init_weight(n_hidden,n_hidden,pre="%s_lstm_f_h_"%prefix)
        self.params += [wz_h]     

        wr_h,b_h = init_weight(n_hidden,n_hidden,pre="%s_lstm_i_h_"%prefix)
        self.params += [wr_h]     

        wc_h,b_h = init_weight(n_hidden,n_hidden,pre="%s_lstm_c_h_"%prefix)
        self.params += [wc_h]     


        #h_t_0 = T.alloc(np.array(0.,dtype=np.float64), x.shape[0], n_hidden)
        #c_t_0 = T.alloc(np.array(0.,dtype=np.float64), x.shape[0], n_hidden)
        h_t_0 = T.alloc(0., x.shape[0], n_hidden)

        #h_t_0 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX))
        #c_t_0 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX))

        h,r = theano.scan(self.recurrent_fn, sequences = [nx,nmask],
                       outputs_info = [h_t_0],
                       non_sequences = [wz_x,wz_h,bz,wr_x,wr_h,br,wc_x,wc_h,bc])

        self.all_hidden = T.transpose(h,axes=(1,0,2))
        self.nn_out = h[-1]

    def recurrent_fn(self,x,mask,h_t_1,wz_x,wz_h,bz,wr_x,wr_h,br,wc_x,wc_h,bc):
        fz = sigmoid(T.dot(h_t_1,wz_h) + T.dot(x,wz_x) + bz)

        fr = sigmoid(T.dot(h_t_1,wr_h) + T.dot(x,wr_x) + br)

        h_new = tanh(T.dot(x,wc_x) + T.dot( (fr*h_t_1) ,wc_h) + bc)

        h_t_this = (1-fz)*h_t_1 + fz*h_new

        h_t = mask[:, None] * h_t_this + (1. - mask)[:, None] * h_t_1

        return h_t


class RNN_batch():
    def __init__(self,n_in,n_hidden,x=T.tensor3("x"),mask=T.matrix("mask"),prefix=""):
         
        self.params = []
        if x is not None:
            self.x = x
        else:
            self.x = T.tensor3("x")

        if mask is not None:
            self.mask = mask
        else:
            self.mask = T.matrix("mask")

        #### 转置 为了进行scan运算 ###
    
        nmask = T.transpose(self.mask,axes=(1,0))
        nx = T.transpose(self.x,axes=(1,0,2))

        w_x,b = init_weight(n_in,n_hidden,pre="%s_lstm_f_x_"%prefix) 
        self.params += [w_x,b]

        w_h,b_h = init_weight(n_hidden,n_hidden,pre="%s_lstm_f_h_"%prefix)
        self.params += [w_h]     

        h_t_0 = T.alloc(0., x.shape[0], n_hidden)

        h,r = theano.scan(self.recurrent_fn, sequences = [nx,nmask],
                       outputs_info = [h_t_0],
                       non_sequences = [w_x,w_h,b])

        self.all_hidden = T.transpose(h,axes=(1,0,2))
        self.nn_out = h[-1]

    def recurrent_fn(self,x,mask,h_t_1,w_x,w_h,b):

        h_t_this = tanh(T.dot(x,w_x) + T.dot(h_t_1,w_h) + b)

        h_t = mask[:, None] * h_t_this + (1. - mask)[:, None] * h_t_1

        return h_t



class RNN():
    def __init__(self,n_in,n_hidden,x=None):
         
        self.params = []
        if x:
            self.x = x
        else:
            self.x = T.matrix("x")

        self.y = T.ivector("y")

        w_in,b_in = init_weight(n_in,n_hidden) 
        self.params += [w_in,b_in]

        w_h,b_h = init_weight(n_hidden,n_hidden)
        self.params += [w_h,b_h]     

        h_t_0 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX))
        h, r = theano.scan(self.recurrent_fn, sequences = self.x,
                       outputs_info = [h_t_0],
                       non_sequences = [w_in ,w_h, b_h])

        self.nn_out = h[-1]
        self.all_hidden = h 


    def recurrent_fn(self,x,h_t_1,w_in,w_h,b):
        h_t = sigmoid(T.dot(h_t_1, w_h) + T.dot(x, w_in) + b)
        return h_t



class RNN_attention():
    def __init__(self,n_in,n_hidden,attention,x=None):
         
        self.params = []
        if x:
            self.x = x
        else:
            self.x = T.matrix("x")

        self.y = T.ivector("y")

        w_in,b_in = init_weight(n_in,n_hidden,pre="aRNN_x_") 
        self.params += [w_in]

        w_h,b_h = init_weight(n_hidden,n_hidden,pre="aRNN_h_")
        self.params += [w_h,b_h]     

        w_attention_x,b_attention = init_weight(n_in,n_hidden,pre="aRNN_attention_x_") 
        self.params += [w_attention_x,b_attention]

        w_attention_a,b_attention_a = init_weight(n_hidden*2,n_hidden,pre="aRNN_attention_a_") 
        self.params += [w_attention_a]

        w_attention_h,b_attention_h = init_weight(n_hidden,n_hidden,pre="aRNN_attention_a_") 
        self.params += [w_attention_h]

        h_t_0 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX))
        h, r = theano.scan(self.recurrent_fn, sequences = self.x,
                       outputs_info = [h_t_0],
                       non_sequences = [w_in ,w_h, b_h, w_attention_x,b_attention,w_attention_a,attention,w_attention_h])

        self.nn_out = h[-1]
        self.all_hidden = h

    def recurrent_fn(self,x,h_t_1,w_in,w_h,b,w_attention_x,b_attention,w_attention_a,attention,w_attention_h):
        h_t_ = tanh(T.dot(h_t_1, w_h) + T.dot(x, w_in) + b)
        #ih = tanh(T.dot(x,w_attention_x)+b_attention + T.dot(attention,w_attention_a) + T.dot(h_t_1,w_attention_h))
        #ih = sigmoid(T.dot(x,w_attention_x)+b_attention + T.dot(attention,w_attention_a) + T.dot(h_t_1,w_attention_h))
        ih = sigmoid(b_attention + T.dot(attention,w_attention_a) + T.dot(h_t_1,w_attention_h) + T.dot(x,w_attention_x))
        #h_t = h_t_ * ih[0]
        h_t = h_t_ * ih
        return h_t

def main():
    r = NetWork(2,2,4,2)
    t = [0,1,0]
    zp_x = [[2,3],[1,2],[2,3]]

    np_x = [[[1,2],[2,3],[3,1]],[[2,3],[1,2],[2,3]],[[3,3],[1,2],[2,3]]]
    npp_x = [[[2,3],[1,2],[2,3]],[[3,3],[1,2],[2,3]]]
    tt = [0,1]

    print r.get_candidate_out(np_x)
    print r.get_candidate_out_inverse(np_x)
    print r.get_zp_out(zp_x,zp_x)
    print r.get_zp_out_inverse(zp_x,zp_x)
    print r.get_sequence_out(np_x)
    print r.get_dot_output(np_x,zp_x,zp_x)
    print r.get_out(np_x,zp_x,zp_x)
    print r.get_cost(np_x,zp_x,zp_x,t)[0]

    print "Train"
    r.train_step(np_x,zp_x,zp_x,t,5)
    r.train_step(np_x,zp_x,zp_x,t,5)
    r.train_step(npp_x,zp_x,zp_x,tt,5)

    print r.get_out(np_x,zp_x,zp_x)
    print r.get_cost(np_x,zp_x,zp_x,t)

    q = list(r.get_out(np_x,zp_x,zp_x)[0][0])
    for num in q:
        print num


    '''
    print r.predict_y(zp_x,zp_x,zp_x)[0][0]
    r.train_step(zp_x,zp_x,zp_x,t,5)
    print r.predict_y(zp_x,zp_x,zp_x)[0][0]
    r.train_step(zp_x,zp_x,zp_x,t,5)
    print r.predict_y(zp_x,zp_x,zp_x)[0][0]
    r.train_step(zp_x,zp_x,zp_x,t,5)
    r.train_step(zp_x,zp_x,zp_x,t,5)
    r.train_step(zp_x,zp_x,zp_x,t,5)
    r.train_step(zp_x,zp_x,zp_x,t,5)
    r.train_step(zp_x,zp_x,zp_x,t,5)
    r.train_step(zp_x,zp_x,zp_x,t,5)
    r.train_step(zp_x,zp_x,zp_x,t,5)
    print r.predict_y(zp_x,zp_x,zp_x)[0][0]
    r.show_para()
    '''

def test_batch():
    x = [[[1,1],[1,1],[1,1]],[[2,2],[2,2],[2,2]],[[3,3],[3,3],[3,3]],[[4,4],[4,4],[4,4]]]
    mask = [[1,0,1],[1,1,1],[0,0,1],[0,1,1]]
    v = [1,1,1]

    #lstm = LSTM_batch(2,3)
    lstm = GRU_batch(2,3)

    vzp = T.vector()
    
    attention = T.sum((softmax(T.sum(vzp*lstm.nn_out,axis=[1]))[0])[:,None]*lstm.nn_out,axis=0)

    f = theano.function(inputs=[lstm.x,lstm.mask],outputs=[lstm.all_hidden])
    ff = theano.function(inputs=[lstm.x,lstm.mask],outputs=[lstm.nn_out])

    fa = theano.function(inputs=[lstm.x,lstm.mask,vzp],outputs=[attention])

    print f(x,mask)
    print ff(x,mask)
    print fa(x,mask,v)


if __name__ == "__main__":
    #main()
    #test()
    test_batch()
