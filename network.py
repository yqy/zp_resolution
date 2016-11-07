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

#对于RNN的神经网络
#一个LSTM for ZP
#一个attention-RNN for NP

'''
Created by qyyin 2015.11.16
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

def init_weight(n_in,n_out,activation_fn=sigmoid,pre="",uni=False,ones=False):
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
    '''
    b_values = np.asarray(
        rng.uniform(
            low=-np.sqrt(1. / np.sqrt(n_out)),
            high=np.sqrt(1. / np.sqrt(n_out)),
            size=(n_out,)
            ),
        dtype=theano.config.floatX
    )
    '''
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

class fakeLayer():
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



class NetWork():
    def __init__(self,n_hidden,embedding_dimention=50):

        ##n_in: sequence lstm 的输入维度
        ##n_hidden: lstm for candi and zp 的隐层维度
        ##n_hidden_sequence: sequence lstm的隐层维度 因为要同zp的结合做dot，所以其维度要是n_hidden的2倍
        ##                   即 n_hidden_sequence = 2 * n_hidden
        self.params = []
        self.zp_x_pre = T.matrix("zp_x_pre")
        self.zp_x_post = T.matrix("zp_x_post")

        zp_nn_pre = LSTM(embedding_dimention,n_hidden,self.zp_x_pre)
        self.params += zp_nn_pre.params
        
        zp_nn_post = LSTM(embedding_dimention,n_hidden,self.zp_x_post)
        self.params += zp_nn_post.params

        self.zp_out = T.concatenate((zp_nn_pre.nn_out,zp_nn_post.nn_out))
        
        self.get_zp_out = theano.function(inputs=[self.zp_x_pre,self.zp_x_post],outputs=[self.zp_out])
    
        ### get sequence output for NP ###
        self.np_x = T.tensor3("np_x")
        self.mask = T.matrix("mask")
    
        self.np_nn = LSTM_batch(embedding_dimention,n_hidden*2,self.np_x,self.mask)
        self.np_out = self.np_nn.nn_out
        self.get_np_out = theano.function(inputs=[self.np_x,self.mask],outputs=[self.np_out])

        
        ### get attention for ZP and NP ###
        dot = self.np_out*self.zp_out 
        self.get_dot = theano.function(inputs=[self.zp_x_pre,self.zp_x_post,self.np_x,self.mask],outputs=[T.sum(dot,axis=[1])])
        

        attention = softmax(T.sum(dot,axis=[1]))[0] 
        
        self.get_attention = theano.function(inputs=[self.zp_x_pre,self.zp_x_post,self.np_x,self.mask],outputs=[attention])
    
        new_zp = T.sum(attention[:,None]*self.np_out,axis=0)

        self.out = attention
        self.get_out = theano.function(inputs=[self.zp_x_pre,self.zp_x_post,self.np_x,self.mask],outputs=[self.out])

        
        l1_norm_squared = sum([(w**2).sum() for w in self.params])
        l2_norm_squared = sum([(abs(w)).sum() for w in self.params])

        lmbda_l1 = 0.0
        #lmbda_l2 = 0.0001
        lmbda_l2 = 0.0

        t = T.bvector()
        #cost = -(T.log((self.out*t).sum()))
        cost = -(T.log((self.out*t).sum()))
        #cost = 1-((self.out*t).sum())

        self.get_cost = theano.function(inputs=[self.zp_x_pre,self.zp_x_post,self.np_x,self.mask,t],outputs=[cost])

        lr = T.scalar()
        #grads = T.grad(cost, self.params)
        #updates = [(param, param-lr*grad)
        #    for param, grad in zip(self.params, grads)]
        
        updates = lasagne.updates.sgd(cost, self.params, lr)
        #updates = lasagne.updates.adadelta(cost, self.params)

        
        self.train_step = theano.function(inputs=[self.zp_x_pre,self.zp_x_post,self.np_x,self.mask,t,lr], outputs=[cost],
            on_unused_input='warn',
            updates=updates)
            #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
            #) 

    def show_para(self):
        for para in self.params:
            print >> sys.stderr, para,para.get_value() 


class LSTM_need():
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


class RNN_attention():
    def __init__(self,n_in,n_hidden,n_out,attention,x=None):
         
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

        #w_attention_x,b_attention = init_weight(n_in,1,pre="aRNN_attention_x_") 
        w_attention_x,b_attention = init_weight(n_in,n_hidden,pre="aRNN_attention_x_") 
        self.params += [w_attention_x,b_attention]
        #self.params += [b_attention]

        #w_attention_a,b_attention_a = init_weight(n_hidden*2,1,pre="aRNN_attention_a_") 
        w_attention_a,b_attention_a = init_weight(n_hidden*2,n_hidden,pre="aRNN_attention_a_") 
        self.params += [w_attention_a]

        #w_attention_h,b_attention_h = init_weight(n_hidden,1,pre="aRNN_attention_a_") 
        w_attention_h,b_attention_h = init_weight(n_hidden,n_hidden,pre="aRNN_attention_a_") 
        self.params += [w_attention_h]

        h_t_0 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX))
        h, r = theano.scan(self.recurrent_fn, sequences = self.x,
                       outputs_info = [h_t_0],
                       non_sequences = [w_in ,w_h, b_h, w_attention_x,b_attention,w_attention_a,attention,w_attention_h])

        #w_out,b_out = init_weight(n_hidden,n_out,pre="aRNN_out_")
        #self.params += [w_out]      

        self.nn_out = h[-1]
        #self.nn_out = T.dot(h[-1],w_out)

    def recurrent_fn(self,x,h_t_1,w_in,w_h,b,w_attention_x,b_attention,w_attention_a,attention,w_attention_h):
        #h_t_ = sigmoid(T.dot(h_t_1, w_h) + T.dot(x, w_in) + b)
        h_t_ = tanh(T.dot(h_t_1, w_h) + T.dot(x, w_in) + b)
        #ih = tanh(T.dot(x,w_attention_x)+b_attention + T.dot(attention,w_attention_a) + T.dot(h_t_1,w_attention_h))
        #ih = sigmoid(T.dot(x,w_attention_x)+b_attention + T.dot(attention,w_attention_a) + T.dot(h_t_1,w_attention_h))
        ih = sigmoid(b_attention + T.dot(attention,w_attention_a) + T.dot(h_t_1,w_attention_h) + T.dot(x,w_attention_x))
        #h_t = h_t_ * ih[0]
        h_t = h_t_ * ih
        return h_t


class aAdd_Layer():
    def __init__(self,n_hidden,x=None):
         
        self.params = []
        if x:
            self.x = x
        else:
            #self.x = T.matrix("x")
            self.x = T.tensor3("x")

        l_in = lasagne.layers.InputLayer(shape=(None, None, 2))
        l_forward_1 = lasagne.layers.LSTMLayer(
        l_in, 3, 
        nonlinearity=lasagne.nonlinearities.tanh)

        l_forward_slice = lasagne.layers.SliceLayer(l_forward_1, -1, 1)

        lstm_out = lasagne.layers.get_output(l_forward_slice)

        self.get_lstm_out = theano.function(inputs=[l_in.input_var],outputs=[lstm_out])
       
 
        #self.give_x = T.matrix("xx")

        #self.add = Add_Layer(3,lstm_out)
 
        h_t_0 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX))
        h, r = theano.scan(self.recurrent_fn, sequences = lstm_out,
                       outputs_info = [h_t_0])

        self.nn_out = h[-1]
        self.out = h

        self.get_out = theano.function(inputs=[l_in.input_var],outputs=[h])

        cost = (self.nn_out[0]).mean()

        all_params = lasagne.layers.get_all_params(l_forward_slice,trainable=True)
        updates = lasagne.updates.adagrad(cost, all_params)
        self.train = theano.function([l_in.input_var], cost, updates=updates, allow_input_downcast=True)


    def recurrent_fn(self,x,h_t_1):
        #self.add.x = x
        #heihei = self.add.nn_out
        #heihei = self.add.get_out(x)
        h_t = x + h_t_1
        return h_t


class Add_Layer():
    def __init__(self,n_hidden,x=None):
         
        self.params = []
        if x:
            self.x = x
        else:
            self.x = T.matrix("x")
        
        h_t_0 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX))
        h, r = theano.scan(self.recurrent_fn, sequences = self.x,
                       outputs_info = [h_t_0])

        self.nn_out = h[-1]
        self.out = h

        self.get_out = theano.function(inputs=[self.x],outputs=[h[-1]])



    def recurrent_fn(self,x,h_t_1):
        h_t = x + h_t_1
        return h_t


class RNN():
    def __init__(self,n_in,n_hidden,n_out,x=None):
         
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

        w_out,b_out = init_weight(n_hidden,n_out)
        self.params += [w_out,b_out]      

        self.nn_out = sigmoid(T.dot(h[-1],w_out) + b_out) 

        self.out = softmax(self.nn_out)

        self.predict_y = theano.function(inputs=[self.x],outputs=[self.out])

        l1_norm_squared = sum([(w**2).sum() for w in self.params])
        l2_norm_squared = sum([(abs(w)).sum() for w in self.params])

        lmbda_l1 = 0.0005
        lmbda_l2 = 0.0

        t = T.bscalar() #标准分类结果
        cost = -(T.log(self.out[0])[t]) +\
            lmbda_l1*l1_norm_squared + lmbda_l2*l2_norm_squared

        self.c = theano.function(inputs=[self.x,t],outputs=[cost])

        lr = T.scalar()
        grads = T.grad(cost, self.params)
        updates = [(param, param-lr*grad)
            for param, grad in zip(self.params, grads)]

        self.train_step = theano.function(inputs=[self.x, t, lr], outputs=[cost],
                            on_unused_input='warn',
                            updates=updates
                            #allow_input_downcast=True
                            #allow_input_downcast=None
                            )

    def recurrent_fn(self,x,h_t_1,w_in,w_h,b):
        h_t = sigmoid(T.dot(h_t_1, w_h) + T.dot(x, w_in) + b)
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

def test():
    #add = Add_Layer(2) 
    #zp_x = [[2,3],[1,2],[2,3]]
    #print add.get_out(zp_x)   
    zp_xx = [[[2,3],[1,2],[2,3]],[[2,3],[1,2],[2,3]],[[2,3],[1,2],[2,3]]]
    zp_xxx = [[[0,0],[2,3],[1,2],[2,3]],[[0,0],[2,3],[1,2],[2,3]],[[0,0],[2,3],[1,2],[2,3]]]

    l_in = lasagne.layers.InputLayer(shape=(None, None, 2))
    l_forward_1 = lasagne.layers.LSTMLayer(
        l_in, 3, 
        nonlinearity=lasagne.nonlinearities.tanh)

    l_forward_slice = lasagne.layers.SliceLayer(l_forward_1, -1, 1)

    out = lasagne.layers.get_output(l_forward_slice)

    f = theano.function(inputs=[l_in.input_var],outputs=[out])

    print f(zp_xx)

    addd = aAdd_Layer(3)
    print addd.get_lstm_out(zp_xx)   
    print addd.get_out(zp_xx)   

    addd.train(zp_xx)
    addd.train(zp_xx)
    print addd.get_out(zp_xx)   

    print addd.get_lstm_out(zp_xxx)   
    print addd.get_out(zp_xxx)   


 
def test_fn(self,x,h_t_1,w_in,w_h,b):
    h_t = sigmoid(T.dot(h_t_1, w_h) + T.dot(x, w_in) + b)
    return h_t
def test_batch():
    x = [[[1,1],[1,1],[1,1]],[[2,2],[2,2],[2,2]],[[3,3],[3,3],[3,3]],[[4,4],[4,4],[4,4]]]
    mask = [[1,0,1],[1,1,1],[0,0,1],[0,1,1]]
    v = [1,1,1]

    lstm = LSTM_batch(2,3)

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