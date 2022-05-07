#!/usr/bin/env python3

import numpy as np
#import tensorflow.keras as keras
import pandas as pd
import sys
import tensorflow.compat.v1 as tf
import timeit
import time
import csv
import datetime
import scipy.io
import scipy.optimize
from scipy import optimize
from scipy.interpolate import CubicSpline
#from statsmodels.tsa.holtwinters import SimpleExpSmothing, Holt
from tqdm import tqdm

tf.disable_v2_behavior()

##########################################################################################################
# load data
df0 = pd.read_csv('world_recovered.csv')
df1 = pd.read_csv('world_confirmed.csv')

##########################################################################################################
# process data

today = '12/11/20' # Update this to include more data 
days = pd.date_range(start='1/22/20',end=today) 
dd = np.arange(len(days))

total_cases = [df1[day.strftime('%-m/%-d/%y')].sum() for day in days] 
total_recov = [df0[day.strftime('%-m/%-d/%y')].sum() for day in days] 

row_r=df0['Country_Region'].tolist().index('US')
total_recov = [df0[day.strftime('%-m/%-d/%y')][row_r] for day in days]

row_c=df1['Country_Region'].tolist().index('US')
total_cases = [df1[day.strftime('%-m/%-d/%y')][row_c] for day in days]

t = np.reshape(dd, [-1])
R = np.reshape(total_recov, [-1])
new_R = R*100/(328.2*10**6) # rescale y-axis

I = np.reshape(total_cases, [-1])
new_I = I*100/(328.2*10**6) # rescale y-axis


# generating more data points for training
nd = 3000
cs1 = CubicSpline(t,new_I)
cs2 = CubicSpline(t,new_R)

Td = np.linspace(0,324,nd)

cs_I = cs1(Td)
cs_R = cs2(Td)


class PINN_PieceBeta:
    # Initialize the class
    def __init__(self, t, I, R, layers1, layers2, M1, M2, M3, b0, xi, lb, ub):
        
        self.lb = lb
        self.ub = ub
        
        self.t = t
        
        self.I = I
        self.R = R
        self.M1 = M1
        self.M2 = M2
        self.M3 = M3
        self.b0 = b0
        self.xi = xi
        
        self.layers1 = layers1
        self.layers2 = layers2
        
        self.weights1, self.biases1 = self.initialize_NN(layers1)
        self.weights2, self.biases2 = self.initialize_NN(layers2)
        
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.gamma = tf.Variable([1.0], dtype=tf.float32)
        #self.xi = tf.Variable([1.0], dtype=tf.float32)
        self.mu = tf.Variable([1.0], dtype=tf.float32)
        
        self.q1 = tf.Variable([1.0], dtype=tf.float32)
        self.q2 = tf.Variable([1.0], dtype=tf.float32)
        self.q3 = tf.Variable([1.0], dtype=tf.float32)
        self.q4 = tf.Variable([1.0], dtype=tf.float32)
        
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.I_tf = tf.placeholder(tf.float32, shape=[None, self.I.shape[1]])
        self.R_tf = tf.placeholder(tf.float32, shape=[None, self.R.shape[1]])
        
                
        
        self.S_pred, self.I_pred, self.J_pred, self.R_pred, self.U_pred = self.net_ASIR(self.t_tf)
        self.p_pred = self.p_net(self.t_tf)
        self.beta_pred = self.betaPiece(self.t_tf)
        
        self.l1, self.l2, self.l3, self.l4, self.l5, self.l6, self.l7 = self.net_l(self.t_tf)
        
        self.loss = tf.reduce_mean(tf.square(self.I_tf - self.I_pred)) + \
            tf.reduce_mean(tf.square(self.I_tf[0]*((1-self.xi)/self.xi) - self.J_pred[0])) + \
            tf.reduce_mean(tf.square(self.R_tf - self.R_pred)) + \
            tf.reduce_mean(tf.square(self.R_tf[0]*((1-self.xi)/self.xi) - self.U_pred[0])) + \
            tf.reduce_mean(tf.square(1.0 - self.p_pred[0])) + \
            tf.reduce_mean(tf.square(self.b0 - self.beta_pred[0])) + \
            tf.reduce_mean(tf.square(self.l1)) + \
            tf.reduce_mean(tf.square(self.l2)) + \
            tf.reduce_mean(tf.square(self.l3)) + \
            tf.reduce_mean(tf.square(self.l4)) + \
            tf.reduce_mean(tf.square(self.l5)) + \
            tf.reduce_mean(tf.square(self.l6)) + \
            tf.reduce_mean(tf.square(self.l7))
        
        self.optimizer = tf.train.AdamOptimizer(1e-3)
        self.train_op = self.optimizer.minimize(self.loss)
        self.loss_log = []
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, t, layers1, weights1, biases1):
        num_layers = len(layers1)
        
        H = 2.0*(t - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights1[l]
            b = biases1[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights1[-1]
        b = biases1[-1]
        #Y = tf.nn.softplus(tf.add(tf.matmul(H, W), b))
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def neural_net2(self, t, layers2, weights2, biases2):
        num_layers = len(layers2)
        
        H = 2.0*(t - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights2[l]
            b = biases2[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights2[-1]
        b = biases2[-1]
        #Y = tf.nn.softplus(tf.add(tf.matmul(H, W), b))
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    
    def net_ASIR(self, t):
        ASIR = self.neural_net(t, self.layers1, self.weights1, self.biases1)
        S = ASIR[:,0:1]
        I = ASIR[:,1:2]
        J = ASIR[:,2:3]
        R = ASIR[:,3:4]
        U = ASIR[:,4:5]

        return S, I, J, R, U

    def p_net(self,t):
        M1 = self.M1
        M2 = self.M2
        M3 = self.M3
        p = self.neural_net2(t, self.layers2, self.weights2, self.biases2)

        q1 = 1.0
        q2 = self.q2
        q3 = self.q3
        q4 = self.q4

        p1 = p[0:M1]
        p2 = p[M1:M2]
        p3 = p[M2:M3]
        p4 = p[M3:]

        p1 = tf.multiply(q1,p1)
        p2 = tf.multiply(q2,p2)
        p3 = tf.multiply(q3,p3)
        p4 = tf.multiply(q4,p4)

        pp =  tf.concat([p1,p2,p3,p4],0)

        return pp

    def betaPiece(self,t):
        b0 =self.b0

        p = self.p_net(t)
        piece_beta = tf.multiply(b0,p)

        return piece_beta
    
    def net_l(self, t):
        NN = 100
        gamma = self.gamma
        xi = self.xi
        mu = self.mu
        
        S, I, J, R, U = self.net_ASIR(t)
        beta = self.betaPiece(t)

        S_t = tf.gradients(S, t)[0]
        I_t = tf.gradients(I, t)[0]
        J_t = tf.gradients(J, t)[0]
        R_t = tf.gradients(R, t)[0]
        U_t = tf.gradients(U, t)[0]
        
        l1 = S_t + beta*(I+J)*S/NN 
        l2 = I_t - xi*beta*(I+J)*S/NN + gamma*I
        l3 = J_t - (1-xi)*beta*(I+J)*S/NN + mu*J
        l4 = R_t - gamma*I 
        l5 = U_t - mu*J 
        
        l6 = NN - (S + I + J + R + U)
        l7 = 0 - (S_t + I_t + J_t + R_t + U_t)
        
        return l1, l2, l3, l4, l5, l6, l7
        
        
    def train(self, nIter):
        tf_dict = {self.t_tf: self.t, self.I_tf: self.I, self.R_tf: self.R}
        start_time = timeit.default_timer()

        for it in tqdm(range(nIter)):
            self.sess.run(self.train_op, tf_dict)
            if it % 100 == 0:
               elapsed = timeit.default_timer() - start_time
               loss_value = self.sess.run(self.loss, tf_dict)
               self.loss_log.append(loss_value)
               q2_value = self.sess.run(self.q2)
               q3_value = self.sess.run(self.q3)
               q4_value = self.sess.run(self.q4)
               #xi_value = self.sess.run(self.xi)
               gamma_value = self.sess.run(self.gamma)
               mu_value = self.sess.run(self.mu)
               start_time = timeit.default_timer()
        
        
        
    def predict(self, t_star):
        tf_dict = {self.t_tf: t_star}
        
        S_star = self.sess.run(self.S_pred, tf_dict)
        I_star = self.sess.run(self.I_pred, tf_dict)
        J_star = self.sess.run(self.J_pred, tf_dict)
        R_star = self.sess.run(self.R_pred, tf_dict)
        U_star = self.sess.run(self.U_pred, tf_dict)
        beta_star = self.sess.run(self.beta_pred, tf_dict)
        
        return S_star, I_star, J_star, R_star, U_star, beta_star
        

##########################################################################################################
# training the network

niter = 40000  # number of Epochs
layers1 = [1, 64, 64, 64, 64, 5]
layers2 = [1, 64, 64, 64, 64, 1]

t_train = Td.flatten()[:,None]
I_train = cs_I.flatten()[:,None]
R_train = cs_R.flatten()[:,None]      
#D_train = cs_D.flatten()[:,None]      

# Doman bounds
lb = t_train.min(0)
ub = t_train.max(0)

model = PINN_PieceBeta(t_train, I_train, R_train, layers1, layers2, 185, 324, 926, 0.279, 0.49, lb, ub)
model.train(niter)

# prediction
S_pred, I_pred, J_pred, R_pred, U_pred, Beta_pred = model.predict(t_train)


mse_train_loss = model.loss_log
rmse_train_loss = np.sqrt(mse_train_loss)
print("rmse_train_loss:",*["%.8f"%(x) for x in rmse_train_loss[0:400]])

# flatten array
T0 = t.flatten()
T1 = t_train.flatten()
I0 = new_I.flatten()
R0 = new_R.flatten()
S1 = S_pred.flatten()
I1 = I_pred.flatten()
J1 = J_pred.flatten()
R1 = R_pred.flatten()
U1 = U_pred.flatten()
Beta1 = Beta_pred.flatten()

# convert float to list
T0 = T0.tolist()
T1 = T1.tolist()
I0 = I0.tolist()
R0 = R0.tolist()
S1 = S1.tolist()
I1 = I1.tolist()
J1 = J1.tolist()
R1 = R1.tolist()
U1 = U1.tolist()
Beta1 = Beta1.tolist()

print("days:",*["%.8f"%(x) for x in T0[0:nd]])
print("genDate:",*["%.8f"%(x) for x in T1[0:nd]])
print("cases:",*["%.8f"%(x) for x in I0[0:nd]])
print("wellness:",*["%.8f"%(x) for x in R0[0:nd]])
print("susceptible:",*["%.8f"%(x) for x in S1[0:nd]])
print("infectd:",*["%.8f"%(x) for x in I1[0:nd]])
print("INFasymp:",*["%.8f"%(x) for x in J1[0:nd]])
print("recoverd:",*["%.8f"%(x) for x in R1[0:nd]])
print("RECasymp:",*["%.8f"%(x) for x in U1[0:nd]])
print("PieceBeta:",*["%.8f"%(x) for x in Beta1[0:nd]])


q2_value = model.sess.run(model.q2)
q3_value = model.sess.run(model.q3)
q4_value = model.sess.run(model.q4)

gamma_value = model.sess.run(model.gamma)
mu_value = model.sess.run(model.mu)


# learned parameters
print("SecondQ:",*["%.8f"%(x) for x in q2_value])
print("ThirdQ:",*["%.8f"%(x) for x in q3_value])
print("FourthQ:",*["%.8f"%(x) for x in q4_value])

print("gamma:",*["%.8f"%(x) for x in gamma_value])
print("mu:",*["%.8f"%(x) for x in mu_value])

##########################################################################################################



