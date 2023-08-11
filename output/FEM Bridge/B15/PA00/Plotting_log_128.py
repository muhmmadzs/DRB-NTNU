# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:49:02 2022

@author: muhamzs
Plotting file for AE Paper 3:
Experimental Data_Set

"""

import os
import warnings
# Dependency imports
from absl import app
from absl import flags
import matplotlib
# matplotlib.use('Agg')
from matplotlib import figure  # pylint: disable=g-import-not-at-top
from matplotlib.backends import backend_agg
import numpy as np
import math
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from absl import flags
from random import random
import scipy.signal
from scipy.signal import find_peaks
from scipy import stats
import matplotlib.pyplot as plt
from scipy.io import savemat
import matplotlib
from sklearn.svm import OneClassSVM
from sklearn.linear_model import SGDOneClassSVM
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
try:
  import seaborn as sns  # pylint: disable=g-import-not-at-top
  HAS_SEABORN = True
except ImportError:
  HAS_SEABORN = False
  
DATASET_INDEX = 8
size=12
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'ytick.labelsize' : size,
    'xtick.labelsize' : size,
    'legend.fontsize' : 12,
    'legend.title_fontsize':12,
    'font.size':size,
    'axes.labelsize': size,
})
plt.rcParams['figure.figsize'] = [3, 4]
dataset_prefix='FEM Bridge/FEM_Acceleration_512'

#*************************************Loading Variables************************************************#
S1_train=np.log(np.load('S1_train.npy'))
S2_train=np.log(np.load('S2_train.npy'))
S3_train=np.log(np.load('S3_train.npy'))
# S4_train=np.log(np.load('S4_train.npy'))
# S5_train=np.log(np.load('S5_train.npy'))
# S6_train=np.log(np.load('S6_train.npy'))

S1_Vali=np.log(np.load('S1_Vali.npy'))
S2_Vali=np.log(np.load('S2_Vali.npy'))
S3_Vali=np.log(np.load('S3_Vali.npy'))
# S4_Vali=np.log(np.load('S4_Vali.npy'))
# S5_Vali=np.log(np.load('S5_Vali.npy'))
# S6_Vali=np.log(np.load('S6_Vali.npy'))



S1_D1=np.log(np.load('S1_D1.npy'))
S2_D1=np.log(np.load('S2_D1.npy'))
S3_D1=np.log(np.load('S3_D1.npy'))
# S4_D1=np.log(np.load('S4_D1.npy'))
# S5_D1=np.log(np.load('S5_D1.npy'))
# S6_D1=np.log(np.load('S6_D1.npy'))

S1_D2=np.log(np.load('S1_D2.npy'))
S2_D2=np.log(np.load('S2_D2.npy'))
S3_D2=np.log(np.load('S3_D2.npy'))
# S4_D2=np.log(np.load('S4_D2.npy'))
# S5_D2=np.log(np.load('S5_D2.npy'))
# S6_D2=np.log(np.load('S6_D2.npy'))

S1_D3=np.log(np.load('S1_D3.npy'))
S2_D3=np.log(np.load('S2_D3.npy'))
S3_D3=np.log(np.load('S3_D3.npy'))
#*************************************Loading Variables************************************************#
# S1_train=(np.load('S1_train.npy'))
# S2_train=(np.load('S2_train.npy'))
# S3_train=(np.load('S3_train.npy'))
# # S4_train=(np.load('S4_train.npy'))
# # S5_train=(np.load('S5_train.npy'))
# # S6_train=(np.load('S6_train.npy'))

# S1_Vali=(np.load('S1_Vali.npy'))
# S2_Vali=(np.load('S2_Vali.npy'))
# S3_Vali=(np.load('S3_Vali.npy'))
# # S4_Vali=(np.load('S4_Vali.npy'))
# # S5_Vali=(np.load('S5_Vali.npy'))
# # S6_Vali=(np.load('S6_Vali.npy'))



# S1_D1=(np.load('S1_D1.npy'))
# S2_D1=(np.load('S2_D1.npy'))
# S3_D1=(np.load('S3_D1.npy'))
# # S4_D1=(np.load('S4_D1.npy'))
# # S5_D1=(np.load('S5_D1.npy'))
# # S6_D1=(np.load('S6_D1.npy'))

# S1_D2=(np.load('S1_D2.npy'))
# S2_D2=(np.load('S2_D2.npy'))
# S3_D2=(np.load('S3_D2.npy'))
# # S4_D2=(np.load('S4_D2.npy'))
# # S5_D2=(np.load('S5_D2.npy'))
# # S6_D2=(np.load('S6_D2.npy'))
# S1_D3=np.load('S1_D3.npy')
# S2_D3=np.load('S2_D3.npy')
# S3_D3=np.load('S3_D3.npy')

#***********************************Fitting Data for training*****************************************#
N=800
Fleet_size=60
S1_train_base=S1_train[0:N]
S1_train_test=S1_train[N:len(S1_train)]

S2_train_base=S2_train[0:N]
S2_train_test=S2_train[N:len(S2_train)]

S3_train_base=S3_train[0:N]
S3_train_test=S3_train[N:len(S3_train)]

# S4_train_base=S4_train[0:N]
# S4_train_test=S4_train[N:len(S4_train)]

# S5_train_base=S5_train[0:N]
# S5_train_test=S5_train[N:len(S5_train)]

# S6_train_base=S6_train[0:N]
# S6_train_test=S6_train[N:len(S6_train)]
# plt.figure()
# res = stats.probplot(S1_train, dist=stats.foldnorm, sparams=1, plot=plt,rvalue=True)
# plt.show
# plt.figure()
# res = stats.probplot(S2_train, dist=stats.foldnorm, sparams=1, plot=plt,rvalue=True)
# plt.show
# plt.figure()
# res = stats.probplot(S3_train, dist=stats.foldnorm, sparams=1, plot=plt,rvalue=True)
# plt.show
# plt.figure()
# res = stats.probplot(S4_train, dist=stats.foldnorm, sparams=1, plot=plt,rvalue=True)
# plt.show
#**************************************
mu_s1, sigma_s1 = scipy.stats.norm.fit(S1_train_base)
mu_s2, sigma_s2 = scipy.stats.norm.fit(S2_train_base)
mu_s3, sigma_s3 = scipy.stats.norm.fit(S3_train_base)
# mu_s4, sigma_s4 = scipy.stats.norm.fit(S4_train_base)
# mu_s5, sigma_s5 = scipy.stats.norm.fit(S5_train_base)
# mu_s6, sigma_s6 = scipy.stats.norm.fit(S6_train_base)


len_Tr_Test=round(len(S1_train_test)/Fleet_size)
len_Vali=round(len(S1_Vali)/Fleet_size)
len_D1=round(len(S1_D1)/Fleet_size)
len_D2=round(len(S1_D2)/Fleet_size)
len_D3=round(len(S1_D3)/Fleet_size)

mean_s1_tr=[]
mean_s2_tr=[]
mean_s3_tr=[]
mean_s4_tr=[] 
mean_s5_tr=[]
mean_s6_tr=[] 
sigma_s1_tr=[]
sigma_s2_tr=[]
sigma_s3_tr=[] 
sigma_s4_tr=[] 
sigma_s5_tr=[] 
sigma_s6_tr=[] 
for i in range(len_Tr_Test):
    mu_s1_tr, si_s1_tr = scipy.stats.norm.fit(S1_train_test[i*Fleet_size:Fleet_size*(i+1)])
    mu_s2_tr, si_s2_tr = scipy.stats.norm.fit(S2_train_test[i*Fleet_size:Fleet_size*(i+1)])
    mu_s3_tr, si_s3_tr = scipy.stats.norm.fit(S3_train_test[i*Fleet_size:Fleet_size*(i+1)])
    # mu_s4_tr, si_s4_tr = scipy.stats.norm.fit(S4_train_test[i*Fleet_size:Fleet_size*(i+1)])
    # mu_s5_tr, si_s5_tr = scipy.stats.norm.fit(S5_train_test[i*Fleet_size:Fleet_size*(i+1)])
    # mu_s6_tr, si_s6_tr = scipy.stats.norm.fit(S6_train_test[i*Fleet_size:Fleet_size*(i+1)])

    mean_s1_tr.append(mu_s1_tr)
    sigma_s1_tr.append(si_s1_tr)
    mean_s2_tr.append(mu_s2_tr)
    sigma_s2_tr.append(si_s2_tr)
    mean_s3_tr.append(mu_s3_tr)
    sigma_s3_tr.append(si_s3_tr)
    # mean_s4_tr.append(mu_s4_tr)
    # sigma_s4_tr.append(si_s4_tr)
    # mean_s5_tr.append(mu_s5_tr)
    # sigma_s5_tr.append(si_s5_tr)
    # mean_s6_tr.append(mu_s6_tr)
    # sigma_s6_tr.append(si_s6_tr)
    
    
mean_s1_val=[]
mean_s2_val=[]
mean_s3_val=[]
mean_s4_val=[] 
mean_s5_val=[]
mean_s6_val=[] 
sigma_s1_val=[]
sigma_s2_val=[]
sigma_s3_val=[] 
sigma_s4_val=[] 
sigma_s5_val=[] 
sigma_s6_val=[] 
for i in range(len_Vali):
    mu_s1_val, si_s1_val = scipy.stats.norm.fit(S1_Vali[i*Fleet_size:Fleet_size*(i+1)])
    mu_s2_val, si_s2_val = scipy.stats.norm.fit(S2_Vali[i*Fleet_size:Fleet_size*(i+1)])
    mu_s3_val, si_s3_val = scipy.stats.norm.fit(S3_Vali[i*Fleet_size:Fleet_size*(i+1)])
    # mu_s4_val, si_s4_val = scipy.stats.norm.fit(S4_Vali[i*Fleet_size:Fleet_size*(i+1)])
    # mu_s5_val, si_s5_val = scipy.stats.norm.fit(S5_Vali[i*Fleet_size:Fleet_size*(i+1)])
    # mu_s6_val, si_s6_val = scipy.stats.norm.fit(S6_Vali[i*Fleet_size:Fleet_size*(i+1)])
    mean_s1_val.append(mu_s1_val)
    sigma_s1_val.append(si_s1_val)
    mean_s2_val.append(mu_s2_val)
    sigma_s2_val.append(si_s2_val)
    mean_s3_val.append(mu_s3_val)
    sigma_s3_val.append(si_s3_val)
    # mean_s4_val.append(mu_s4_val)
    # sigma_s4_val.append(si_s4_val)
    # mean_s5_val.append(mu_s5_val)
    # sigma_s5_val.append(si_s5_val)
    # mean_s6_val.append(mu_s6_val)
    # sigma_s6_val.append(si_s6_val)

mean_s1_d1=[]
mean_s2_d1=[]
mean_s3_d1=[]
mean_s4_d1=[] 
mean_s5_d1=[]
mean_s6_d1=[] 
sigma_s1_d1=[]
sigma_s2_d1=[]
sigma_s3_d1=[] 
sigma_s4_d1=[] 
sigma_s5_d1=[] 
sigma_s6_d1=[] 
for i in range(len_D1):
    mu_s1_d1, si_s1_d1 = scipy.stats.norm.fit(S1_D1[i*Fleet_size:Fleet_size*(i+1)])
    mu_s2_d1, si_s2_d1 = scipy.stats.norm.fit(S2_D1[i*Fleet_size:Fleet_size*(i+1)])
    mu_s3_d1, si_s3_d1 = scipy.stats.norm.fit(S3_D1[i*Fleet_size:Fleet_size*(i+1)])
    # mu_s4_d1, si_s4_d1 = scipy.stats.norm.fit(S4_D1[i*Fleet_size:Fleet_size*(i+1)])
    # mu_s5_d1, si_s5_d1 = scipy.stats.norm.fit(S5_D1[i*Fleet_size:Fleet_size*(i+1)])
    # mu_s6_d1, si_s6_d1 = scipy.stats.norm.fit(S6_D1[i*Fleet_size:Fleet_size*(i+1)])
    mean_s1_d1.append(mu_s1_d1)
    sigma_s1_d1.append(si_s1_d1)
    mean_s2_d1.append(mu_s2_d1)
    sigma_s2_d1.append(si_s2_d1)
    mean_s3_d1.append(mu_s3_d1)
    sigma_s3_d1.append(si_s3_d1)
    # mean_s4_d1.append(mu_s4_d1)
    # sigma_s4_d1.append(si_s4_d1)
    # mean_s5_d1.append(mu_s5_d1)
    # sigma_s5_d1.append(si_s5_d1)
    # mean_s6_d1.append(mu_s6_d1)
    # sigma_s6_d1.append(si_s6_d1)   
    
    
mean_s1_d2=[]
mean_s2_d2=[]
mean_s3_d2=[]
mean_s4_d2=[]
mean_s5_d2=[]
mean_s6_d2=[]  
sigma_s1_d2=[]
sigma_s2_d2=[]
sigma_s3_d2=[] 
sigma_s4_d2=[] 
sigma_s5_d2=[] 
sigma_s6_d2=[] 
for i in range(len_D2):
    mu_s1_d2, si_s1_d2 = scipy.stats.norm.fit(S1_D2[i*Fleet_size:Fleet_size*(i+1)])
    mu_s2_d2, si_s2_d2 = scipy.stats.norm.fit(S2_D2[i*Fleet_size:Fleet_size*(i+1)])
    mu_s3_d2, si_s3_d2 = scipy.stats.norm.fit(S3_D2[i*Fleet_size:Fleet_size*(i+1)])
    # mu_s4_d2, si_s4_d2 = scipy.stats.norm.fit(S4_D2[i*Fleet_size:Fleet_size*(i+1)])
    # mu_s5_d2, si_s5_d2 = scipy.stats.norm.fit(S5_D2[i*Fleet_size:Fleet_size*(i+1)])
    # mu_s6_d2, si_s6_d2 = scipy.stats.norm.fit(S6_D2[i*Fleet_size:Fleet_size*(i+1)])
    mean_s1_d2.append(mu_s1_d2)
    sigma_s1_d2.append(si_s1_d2)
    mean_s2_d2.append(mu_s2_d2)
    sigma_s2_d2.append(si_s2_d2)
    mean_s3_d2.append(mu_s3_d2)
    sigma_s3_d2.append(si_s3_d2)
#     mean_s4_d2.append(mu_s4_d2)
#     sigma_s4_d2.append(si_s4_d2)  
#     mean_s5_d2.append(mu_s5_d2)
#     sigma_s5_d2.append(si_s5_d2)
#     mean_s6_d2.append(mu_s6_d2)
#     sigma_s6_d2.append(si_s6_d2)  
# # plt.figure()
# plt.plot(mean_s3_tr)
# plt.plot(mean_s3_val)
# plt.plot(mean_s3_d1)
# plt.plot(mean_s3_d2)

# plt.figure()
# plt.plot(sigma_s3_tr)
# plt.plot(sigma_s3_val)
# plt.plot(sigma_s3_d1)
# plt.plot(sigma_s3_d2)

mean_s1_d3=[]
mean_s2_d3=[]
mean_s3_d3=[]
mean_s4_d3=[]
mean_s5_d3=[]
mean_s6_d3=[]  
sigma_s1_d3=[]
sigma_s2_d3=[]
sigma_s3_d3=[] 
sigma_s4_d3=[] 
sigma_s5_d3=[] 
sigma_s6_d3=[] 
for i in range(len_D3):
    mu_s1_d3, si_s1_d3 = scipy.stats.norm.fit(S1_D3[i*Fleet_size:Fleet_size*(i+1)])
    mu_s2_d3, si_s2_d3 = scipy.stats.norm.fit(S2_D3[i*Fleet_size:Fleet_size*(i+1)])
    mu_s3_d3, si_s3_d3 = scipy.stats.norm.fit(S3_D3[i*Fleet_size:Fleet_size*(i+1)])
    # mu_s4_d3, si_s4_d3 = scipy.stats.norm.fit(S4_D3[i*Fleet_size:Fleet_size*(i+1)])
    # mu_s5_d3, si_s5_d3 = scipy.stats.norm.fit(S5_D3[i*Fleet_size:Fleet_size*(i+1)])
    # mu_s6_d3, si_s6_d3 = scipy.stats.norm.fit(S6_D3[i*Fleet_size:Fleet_size*(i+1)])
    mean_s1_d3.append(mu_s1_d3)
    sigma_s1_d3.append(si_s1_d3)
    mean_s2_d3.append(mu_s2_d3)
    sigma_s2_d3.append(si_s2_d3)
    mean_s3_d3.append(mu_s3_d3)
    sigma_s3_d3.append(si_s3_d3)
#     mean_s4_d3.append(mu_s4_d3)
#     sigma_s4_d3.append(si_s4_d3)  
#     mean_s5_d3.append(mu_s5_d3)
#     sigma_s5_d3.append(si_s5_d3)
#     mean_s6_d3.append(mu_s6_d3)
#     sigma_s6_d3.append(si_s6_d3)  
# # plt.figure()
# plt.plot(mean_s3_tr)
# plt.plot(mean_s3_val)
# plt.plot(mean_s3_d1)
# plt.plot(mean_s3_d3)

# plt.figure()
# plt.plot(sigma_s3_tr)
# plt.plot(sigma_s3_val)
# plt.plot(sigma_s3_d1)
# plt.plot(sigma_s3_d3)
#***************************************************KL Divergence Calulation***************************#

def kl_mvn(m0, S0, m1, S1):
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    

    From wikipedia
    KL( (m0, S0) || (m1, S1))
         = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| + 
                  (m1 - m0)^T S1^{-1} (m1 - m0) - N )
    """
    # store inv diag covariance of S1 and diff between means
    N = m0.shape[0]
    iS1 = np.linalg.inv(S1)
    diff = m1 - m0

    # kl is made of three terms
    tr_term   = np.trace(iS1 @ S0)
    det_term  = np.log(np.linalg.det(S1)/np.linalg.det(S0)) #np.sum(np.log(S1)) - np.sum(np.log(S0))
    quad_term = diff.T @ np.linalg.inv(S1) @ diff #np.sum( (diff*diff) * iS1, axis=1)
    #print(tr_term,det_term,quad_term)
    kl=.5 * (tr_term + det_term + quad_term - N)
    
    return  kl

mean_train=np.array([mu_s1,mu_s2,mu_s3])
Sigma_train=np.array([sigma_s1,sigma_s2,sigma_s3])
Sigma_train=np.diag(Sigma_train)
print(Sigma_train)
KL_Train=[]
for i in range(len(mean_s1_tr)):
    mean_kl=np.array([mean_s1_tr[i],mean_s2_tr[i],mean_s3_tr[i]])
    sigma_kl1=np.array([sigma_s1_tr[i],sigma_s2_tr[i],sigma_s3_tr[i]])
    sigma_kl=np.diag(sigma_kl1)
    kl1=np.log(kl_mvn(mean_train, Sigma_train ,mean_kl, sigma_kl)+math.exp(1))-1

    #kl1=(np.log(kl_mvn(mean_train, Sigma_train ,mean_kl, sigma_kl))/(math.exp(1.5))+math.exp(1))-1

    KL_Train.append(kl1)
print(sigma_kl)    
print(mean_kl) 
KL_Vali=[]
for i in range(len(mean_s1_val)):
    mean_kl=np.array([mean_s1_val[i],mean_s2_val[i],mean_s3_val[i]])
    sigma_kl2=np.array([sigma_s1_val[i],sigma_s2_val[i],sigma_s3_val[i]])
    sigma_kl=np.diag(sigma_kl2)
    kl2=np.log(kl_mvn(mean_train, Sigma_train ,mean_kl, sigma_kl)+math.exp(1))-1
    
    #kl2=(np.log(kl_mvn(mean_train, Sigma_train ,mean_kl, sigma_kl))/(math.exp(1.5))+math.exp(1))-1

    KL_Vali.append(kl2)    
print(sigma_kl)    
print(mean_kl) 
    
KL_D1=[]
for i in range(len(mean_s1_d1)):
    mean_kl=np.array([mean_s1_d1[i],mean_s3_d1[i],mean_s3_d1[i]])
    sigma_kl3=np.array([sigma_s1_d1[i],sigma_s2_d1[i],sigma_s3_d1[i]])
    sigma_kl=np.diag(sigma_kl3)
    kl3=np.log(kl_mvn(mean_train, Sigma_train ,mean_kl, sigma_kl)+math.exp(1))-1
    
   # kl3=(np.log(kl_mvn(mean_train, Sigma_train ,mean_kl, sigma_kl))/(math.exp(1.5))+math.exp(1))-1
    KL_D1.append(kl3)    
print(sigma_kl)    
print(mean_kl) 

KL_D2=[]
for i in range(len(mean_s1_d2)):
    mean_kl=np.array([mean_s1_d2[i],mean_s2_d2[i],mean_s3_d2[i]])
    sigma_kl4=np.array([sigma_s1_d2[i],sigma_s2_d2[i],sigma_s3_d2[i]])
    sigma_kl=np.diag(sigma_kl4)
    kl4=np.log(kl_mvn(mean_train, Sigma_train ,mean_kl, sigma_kl)+math.exp(1))-1
    
    #kl4=(np.log(kl_mvn(mean_train, Sigma_train ,mean_kl, sigma_kl))/(math.exp(1.5))+math.exp(1))-1
    KL_D2.append(kl4)     
print(sigma_kl)    
print(mean_kl) 

KL_D3=[]
for i in range(len(mean_s1_d3)):
    mean_kl=np.array([mean_s1_d3[i],mean_s2_d3[i],mean_s3_d3[i]])
    sigma_kl5=np.array([sigma_s1_d3[i],sigma_s2_d3[i],sigma_s3_d3[i]])
    sigma_kl=np.diag(sigma_kl5)
    kl5=np.log(kl_mvn(mean_train, Sigma_train ,mean_kl, sigma_kl)+math.exp(1))-1

   # kl5=(np.log(kl_mvn(mean_train, Sigma_train ,mean_kl, sigma_kl))/(math.exp(1.5))+math.exp(1))-1
    KL_D3.append(kl5)     
print(sigma_kl)    
print(mean_kl) 
#***************************************Prob**************************************#



#********************************************************************************
def ewma_vectorized(data, alpha, offset=None, dtype=None, order='C', out=None):
    """
    Calculates the exponential moving average over a vector.
    Will fail for large inputs.
    :param data: Input data
    :param alpha: scalar float in range (0,1)
        The alpha parameter for the moving average.
    :param offset: optional
        The offset for the moving average, scalar. Defaults to data[0].
    :param dtype: optional
        Data type used for calculations. Defaults to float64 unless
        data.dtype is float32, then it will use float32.
    :param order: {'C', 'F', 'A'}, optional
        Order to use when flattening the data. Defaults to 'C'.
    :param out: ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        the same shape as the input. If not provided or `None`,
        a freshly-allocated array is returned.
    """
    data = np.array(data, copy=False)

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64
    else:
        dtype = np.dtype(dtype)

    if data.ndim > 1:
        # flatten input
        data = data.reshape(-1, order)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    if data.size < 1:
        # empty input, return empty array
        return out

    if offset is None:
        offset = data[0]

    alpha = np.array(alpha, copy=False).astype(dtype, copy=False)

    # scaling_factors -> 0 as len(data) gets large
    # this leads to divide-by-zeros below
    scaling_factors = np.power(1. - alpha, np.arange(data.size + 1, dtype=dtype),
                               dtype=dtype)
    # create cumulative sum array
    np.multiply(data, (alpha * scaling_factors[-2]) / scaling_factors[:-1],
                dtype=dtype, out=out)
    np.cumsum(out, dtype=dtype, out=out)

    # cumsums / scaling
    out /= scaling_factors[-2::-1]

    if offset != 0:
        offset = np.array(offset, copy=False).astype(dtype, copy=False)
        # add offsets
        out += offset * scaling_factors[1:]

    return out
#*******************************************************************************#
plt.figure()
index1=np.linspace(0, len(KL_Train),len(KL_Train))
index2=np.linspace(len(index1)+1, len(KL_Vali)+len(index1),len(KL_Vali))
index3=np.linspace(index2[-1]+1, len(KL_D1)+index2[-1],len(KL_D1))
index4=np.linspace(index3[-1]+1, len(KL_D2)+index3[-1],len(KL_D2))
index5=np.linspace(index4[-1]+1, len(KL_D3)+index4[-1],len(KL_D3))
EW_Train=ewma_vectorized(KL_Train,0.05)
EW_Vali=ewma_vectorized(KL_Vali,0.05)
EW_D1=ewma_vectorized(KL_D1,0.05)
EW_D2=ewma_vectorized(KL_D2,0.05)
EW_D3=ewma_vectorized(KL_D2,0.05)
#********************************************EWMA******************************#


plt.scatter(index1,EW_Train, label='Training')
plt.scatter(index2,EW_Vali,label='Validation')
plt.scatter(index3,EW_D1,label='D1')
plt.scatter(index4,EW_D2,label='D2',c ="k")
plt.scatter(index5,EW_D3,label='D3',c ="k")
y_mean_ew = np.mean(EW_Train)
y_std_ew = np.std(EW_Train)
mean_line = plt.hlines(y=y_mean_ew, xmin=index1[0], xmax=index4[-1], colors='r', linestyles='--', lw=.5, label='Mean')
mean_line = plt.hlines(y=[y_mean_ew-2*y_std_ew, y_mean_ew+2*y_std_ew], xmin=index1[0], xmax=index4[-1], colors='k', linestyles='--', lw=0.5, label='(+-2)Std Dev')
plt.yscale("log")
plt.title('EWMA win =256')
# plt.ylim(0,2)
plt.legend()
plt.ylabel('DI')
plt.xlabel('Number of Inspection')

#******************************************Normal Plot************************ #

plt.figure()
plt.scatter(index1, KL_Train, label='Training')
plt.scatter(index2, KL_Vali,label='Validation')
plt.scatter(index3, KL_D1,label='D1')
plt.scatter(index4, KL_D2,label='D2',c ="k")
plt.scatter(index5, KL_D3,label='D3',c ="k")

y_mean = np.mean(KL_Train)
y_KL_Vali = np.mean(KL_Vali)
y_KL_D1 = np.mean(KL_D1)
y_KL_D2 = np.mean(KL_D2)
y_KL_D3 = np.mean(KL_D3)

y_std = np.std(KL_Train)
mean_line = plt.hlines(y=y_mean, xmin=index1[0], xmax=index1[-1], colors='k', linestyles='--', lw=.5, label='Mean')
mean_line = plt.hlines(y=y_KL_Vali, xmin=index2[0], xmax=index2[-1], colors='k', linestyles='--', lw=.5 )
mean_line = plt.hlines(y=y_KL_D1, xmin=index3[0], xmax=index3[-1], colors='k', linestyles='--', lw=.5)
mean_line = plt.hlines(y=y_KL_D2, xmin=index4[0], xmax=index4[-1], colors='k', linestyles='--', lw=.5)
mean_line = plt.hlines(y=y_KL_D3, xmin=index5[0], xmax=index5[-1], colors='k', linestyles='--', lw=.5)

# mean_line = plt.hlines(y=[y_mean-2*y_std, y_mean+2*y_std], xmin=index1[0], xmax=index4[-1], colors='k', linestyles='--', lw=0.5, label='(+-2)Std Dev')
# plt.yscale("log")
# plt.title('DI win =256')
# plt.ylim(0,2)
plt.legend()
plt.ylabel('D1')
plt.xlabel('Number of Inspection')
plt.show()

#*****************************************************
index11=np.linspace(0, len(KL_D1),len(KL_D1))
index22=np.linspace(len(index11)+1, len(KL_D2)+len(index11),len(KL_D2))
index33=np.linspace(index22[-1]+1, len(KL_D3)+index22[-1],len(KL_D3))

plt.figure()
plt.scatter(index11, KL_D1, label='Healthy')
plt.scatter(index22, KL_D2,label='D0')
plt.scatter(index33, KL_D3,label='D1')
y_KL_D1 = np.mean(KL_D1)
y_KL_D2 = np.mean(KL_D2)
y_KL_D3 = np.mean(KL_D3)
mean_line = plt.hlines(y=y_KL_D1, xmin=index11[0], xmax=index11[-1], colors='k', linestyles='--', lw=.5, label='Mean')
mean_line = plt.hlines(y=y_KL_D2, xmin=index22[0], xmax=index22[-1], colors='k', linestyles='--', lw=.5)
mean_line = plt.hlines(y=y_KL_D3, xmin=index33[0], xmax=index33[-1], colors='k', linestyles='--', lw=.5)

# plt.axvline(x =index11[-1], color = 'k')
# plt.axvline(x =index22[-1], color = 'k', linestyles='--')
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend()
plt.ylabel('DI')
plt.xlabel('Number of Inspection')

plt.tight_layout()
plt.show()
#****************************************Box plot*******************************************#
data = [KL_D1, KL_D2,KL_D3]
labels=['DM00' ,'DM20' ,'DM40']
c = "tan"
fig1, ax1 = plt.subplots()
bplot=ax1.boxplot(data,notch=False,  # notch shape
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labels,
                     medianprops=dict(color='red'),
                     boxprops=dict(facecolor=c, color='red'),
                     capprops=dict(color='k'),
                     whiskerprops=dict(color='k'),
                     flierprops=dict(color='k', markeredgecolor='k'))

# colors = ['lightblue', 'lightblue', 'lightblue']
# for patch, color in zip(bplot['boxes'], colors):
#     patch.set_facecolor(color)
ax1.set_ylabel('Damage Index (DI)')
ax1.set_xlabel('Damage cases')
#plt.savefig('Fig_NEW.png', dpi = 1200 ,bbox_inches='tight')
#ax1.set_ylim([0.02, 0.16])
#plt.yscale("log")
#****************************************Roc************************************************
y_pre_train=[]
y_pre_VaLi=[]
y_pre_D1=[]
y_pre_D2=[]
for i in range(len(KL_Train)):
    
    if KL_Train[i]>(y_mean+2*y_std):
        y_predict=1
    else:
        y_predict=0
    y_pre_train.append(y_predict)
    
for i in range(len(KL_Vali)):
    
    if KL_Vali[i]>(y_mean+2*y_std):
        y_predict=1
    else:
        y_predict=0
    y_pre_VaLi.append(y_predict)
    
for i in range(len(KL_D1)):
    
    if KL_D1[i]>(y_mean+2*y_std):
        y_predict=1
    else:
        y_predict=0
    y_pre_D1.append(y_predict)

for i in range(len(KL_D2)):
    
    if KL_D2[i]>(y_mean+2*y_std):
        y_predict=1
    else:
        y_predict=0
    y_pre_D2.append(y_predict)
#************************************ fUNCTION **************************************#
def ewmaa(X,alpha=0.3, coefficient=3):
     """
     Predict if a particular sample is an outlier or not.
     :param X: the time series to detect of
     :param type X: pandas.Series
     :return: 1 denotes normal, 0 denotes abnormal
     """
     
     s = [X[0]]
     #s = [0]

     for i in range(1, len(X)):
         # if i<15:
         #     temp= X[i]
         #     s.append(temp)
         # else:   
        temp = alpha * X[i] + (1 - alpha) * s[-1]
        s.append(temp)
     s_avg = np.mean(s)
     sigma = np.sqrt(np.var(s))
     ucl=[]
     lcl=[]
     for i in range(1, len(X)):
         ucl_temp = s_avg + coefficient * sigma * np.sqrt((alpha / (2 - alpha))*(1-(1-alpha)**(2*i)))
         lcl_temp = s_avg - coefficient * sigma * np.sqrt((alpha / (2 - alpha))*(1-(1-alpha)**(2*i)))
         ucl.append(ucl_temp)
         lcl.append(lcl_temp)
     # if s[-1] > ucl or s[-1] < lcl:
     #     return 0
     # return 1
     return s,ucl,lcl,s_avg,sigma
#************************************ outliers************************************************#
#Tukey's method
def tukeys_method(df):
    #Takes two parameters: dataframe & variable of interest as string
    q = np.quantile(df, q = np.arange(0.25, 1, 0.25))
    print(q)
   # q3 = np.quantile(df, q = np.arange(0.25, 1, 0.25))
    
    q1=q[0]
    q3=q[2]
    print(q1, q3)
    iqr = q3-q1
    print(iqr)
    inner_fence = 1.5*iqr
    outer_fence = 3*iqr
    
    #inner fence lower and upper end
    inner_fence_le = q1-inner_fence
    inner_fence_ue = q3+inner_fence
    
    #outer fence lower and upper end
    outer_fence_le = q1-outer_fence
    outer_fence_ue = q3+outer_fence
    
    outliers_prob = []
    outliers_poss = []
    for index, x in enumerate(df):
        if x <= outer_fence_le or x >= outer_fence_ue:
            outliers_prob.append(index)
    for index, x in enumerate(df):
        if x <= inner_fence_le or x >= inner_fence_ue:
            outliers_poss.append(index)
    return outer_fence_le, outer_fence_ue, inner_fence_le, inner_fence_ue 


#************************************ EWMA_2 **************************************************
Out_Ewma=[]

outer_fence_le,outer_fence_ue, inner_fence_le, inner_fence_ue  = tukeys_method(KL_Train)

print(outer_fence_le, outer_fence_ue, inner_fence_le, inner_fence_ue)

# Out_Ewma=KL_Train+KL_Vali+KL_D1+KL_D2
n=outer_fence_ue
n=40
Out_Ewma_tr=[]
Out_Ewma_tr=KL_Train
filtered = filter(lambda num: num < n, Out_Ewma_tr)
Out_Ewma_tr=list(filtered)

Out_Ewma_Vali=[]
Out_Ewma_Vali=KL_Vali
filtered = filter(lambda num: num < n, Out_Ewma_Vali)
Out_Ewma_Vali=list(filtered)

Out_Ewma_D1=[]
Out_Ewma_D1=KL_D1

filtered = filter(lambda num: num < n, Out_Ewma_D1)
Out_Ewma_D1=list(filtered)


Out_Ewma_D2=[]
Out_Ewma_D2=KL_D2

filtered = filter(lambda num: num < n, Out_Ewma_D2)
Out_Ewma_D2=list(filtered)

Out_Ewma=Out_Ewma_tr+Out_Ewma_Vali+Out_Ewma_D1+Out_Ewma_D2

a = Out_Ewma
filtered = filter(lambda num: num < n, a)
Out_Ewma_2=list(filtered)

plt.figure()
plt.plot(Out_Ewma_2)
plt.figure()
plt.plot(Out_Ewma)


index1=np.linspace(0, len(Out_Ewma_tr),len(Out_Ewma_tr))
index2=np.linspace(len(index1)+1, len(Out_Ewma_Vali)+len(index1),len(Out_Ewma_Vali))
index3=np.linspace(index2[-1]+1, len(Out_Ewma_D1)+index2[-1],len(Out_Ewma_D1))
index4=np.linspace(index3[-1]+1, len(Out_Ewma_D2)+index3[-1],len(Out_Ewma_D2))



EW,_,_,_,_=ewmaa(Out_Ewma, 0.03,coefficient=50)
_,ucl,lcl,s_avg,sigma=ewmaa(Out_Ewma_tr, 0.03,coefficient=50)

plt.figure()
plt.plot(EW)

ind=np.linspace(0, len(EW)-1,len(EW)-1)
ucl = np.hstack((ucl, np.tile(ucl[-1], (len(Out_Ewma_Vali)+len(Out_Ewma_D1)+len(Out_Ewma_D2)))))
lcl = np.hstack((lcl, np.tile(lcl[-1], (len(Out_Ewma_Vali)+len(Out_Ewma_D1)+len(Out_Ewma_D2)))))

"""
plt.figure()
# plt.scatter(ind, EW, label='Training')
plt.scatter(index1, EW[0:len(KL_Train)], label='Training')
plt.scatter(index2, EW[len(KL_Train):len(KL_Train)+len(KL_Vali)],label='Validation')
plt.scatter(index3, EW[len(KL_Train)+len(KL_Vali):len(KL_D1)+len(KL_Train)+len(KL_Vali)],label='D1')
plt.scatter(index4, EW[len(KL_D1)+len(KL_Train)+len(KL_Vali):len(KL_D1)+len(KL_Train)+len(KL_Vali)+len(KL_D2)],label='D2',c ="k")
plt.plot(ind,ucl,linestyle='--',c ="k", lw=.5,label='UCL')
plt.plot(ind,lcl,linestyle='--',c ="k", lw=.5,label='LCL')
plt.fill_between(ind, ucl, lcl,alpha=0.2)
plt.fill_between(ind, ucl, lcl,alpha=0.2)
"""
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
#plt.figure()
# plt.scatter(ind, EW, label='Training')
ax1.scatter(index1, EW[0:len(Out_Ewma_tr)], label='Training')
ax1.scatter(index2, EW[len(Out_Ewma_tr):len(Out_Ewma_tr)+len(Out_Ewma_Vali)],label='Validation')
ax1.scatter(index3, EW[len(Out_Ewma_tr)+len(Out_Ewma_Vali):len(Out_Ewma_D1)+len(Out_Ewma_tr)+len(Out_Ewma_Vali)],label='D1')
ax1.scatter(index4, EW[len(Out_Ewma_D1)+len(Out_Ewma_tr)+len(Out_Ewma_Vali):len(Out_Ewma_D1)+len(Out_Ewma_tr)+len(Out_Ewma_Vali)+len(Out_Ewma_D2)],label='D2',c ="k")
ax1.plot(ind,ucl,linestyle='--',c ="k", lw=.5,label='UCL')
ax1.plot(ind,lcl,linestyle='--',c ="k", lw=.5,label='LCL')
ax1.fill_between(ind, ucl, lcl,alpha=0.02)
ax1.fill_between(ind, ucl, lcl,alpha=0.02)
# y_mean = np.mean(EW[0:len(KL_Train)])
# y_mean = np.mean(EW[0:len(KL_Train)])

#y_std = np.std(EW[0:len(KL_Train)])
#mean_line = plt.hlines(y=y_mean, xmin=index1[0], xmax=index4[-1], colors='r', linestyles='--', lw=.5, label='Mean')
#mean_line = plt.hlines(y=[y_mean-2*y_std, y_mean+2*y_std], xmin=index1[0], xmax=index4[-1], colors='k', linestyles='--', lw=0.5, label='(+-2)Std Dev')
# plt.yscale("log")
# plt.title('Accleration Signals')
# ax1.set_ylim(0.03,0.06)
plt.xlim(0,len(EW))
ax1.legend()
#plt.ylabel('D1')
# ax1.xlabel('Number of Inspection')
#plt.show()


#************************************ Plotting Without Label 2 **************************************************
"""
 
index1=np.linspace(0, len(KL_Train),len(KL_Train))
index2=np.linspace(len(index1)+1, len(KL_Vali)+len(index1),len(KL_Vali))
index3=np.linspace(index2[-1]+1, len(KL_D2)+index2[-1],len(KL_D2))
# index4=np.linspace(index3[-1]+1, len(KL_D2)+index3[-1],len(KL_D2))

Out_Ewma=[]

Out_Ewma=KL_Train+KL_Vali+KL_D2
Out_Ewma = np.array(Out_Ewma)
Out_Ewma_2=KL_Train
Out_Ewma_2 = np.array(Out_Ewma_2)
EW,_,_,_,_=ewmaa(Out_Ewma, 0.02,coefficient=24)
EW221,ucl,lcl,s_avg,sigma=ewmaa(Out_Ewma_2, 0.02,coefficient=24)

ind=np.linspace(0, len(Out_Ewma)-1,len(Out_Ewma)-1)
ucl = np.hstack((ucl, np.tile(ucl[-1], (len(KL_Vali)+len(KL_D2)))))
lcl = np.hstack((lcl, np.tile(lcl[-1], (len(KL_Vali)+len(KL_D2)))))

plt.figure()
# plt.scatter(ind, EW, label='Training')
# plt.style.use('grayscale')

plt.scatter(index1, EW[0:len(KL_Train)], label='Training')
plt.scatter(index2, EW[len(KL_Train):len(KL_Train)+len(KL_Vali)],label='Validation')
plt.scatter(index3, EW[len(KL_Train)+len(KL_Vali):len(KL_Train)+len(KL_Vali)+len(KL_D2)],label='D2')
plt.plot(ind,ucl,linestyle='--',c ="k", lw=.5,label='UCL')
plt.plot(ind,lcl,linestyle='--',c ="k", lw=.5,label='LCL')
plt.fill_between(ind, ucl, lcl,alpha=0.2)
y_mean = np.mean(EW[0:len(KL_Train)])
y_mean = np.mean(EW[0:len(KL_Train)])

y_std = np.std(EW[0:len(KL_Train)])

plt.title('With Outlier')
# plt.ylim(0,0.22)
plt.xlim(0,len(EW))
plt.legend()
plt.ylabel('D1')
plt.xlabel('Number of Inspection')
plt.show()
"""
#************************************ Plotting Without Label 2 **************************************************
"""
 
index1=np.linspace(0, len(Out_Ewma_tr),len(Out_Ewma_tr))
index2=np.linspace(len(index1)+1, len(Out_Ewma_Vali)+len(index1),len(Out_Ewma_Vali))
index3=np.linspace(index2[-1]+1, len(Out_Ewma_D2)+index2[-1],len(Out_Ewma_D2))
# index4=np.linspace(index3[-1]+1, len(KL_D2)+index3[-1],len(KL_D2))

Out_Ewma_New=[]
Out_Ewma_New=Out_Ewma_tr+Out_Ewma_Vali+Out_Ewma_D2
Out_Ewma_2=[]
Out_Ewma_2=Out_Ewma_tr
EW,_,_,_,_=ewmaa(Out_Ewma_New, 0.01,coefficient=10)
EW221,ucl,lcl,s_avg,sigma=ewmaa(Out_Ewma_2, 0.01,coefficient=10)

ind=np.linspace(0, len(Out_Ewma_New)-1,len(Out_Ewma_New)-1)
ucl = np.hstack((ucl, np.tile(ucl[-1], (len(Out_Ewma_Vali)+len(Out_Ewma_D2)))))
lcl = np.hstack((lcl, np.tile(lcl[-1], (len(Out_Ewma_Vali)+len(Out_Ewma_D2)))))

plt.figure()
# plt.scatter(ind, EW, label='Training')
# plt.style.use('grayscale')

plt.scatter(index1, EW[0:len(Out_Ewma_tr)], label='Training')
plt.scatter(index2, EW[len(Out_Ewma_tr):len(Out_Ewma_tr)+len(Out_Ewma_Vali)],label='Validation')
plt.scatter(index3, EW[len(Out_Ewma_tr)+len(Out_Ewma_Vali):len(Out_Ewma_tr)+len(Out_Ewma_Vali)+len(Out_Ewma_D2)],label='D2')
plt.plot(ind,ucl,linestyle='--',c ="k", lw=.5,label='UCL')
plt.plot(ind,lcl,linestyle='--',c ="k", lw=.5,label='LCL')
plt.fill_between(ind, ucl, lcl,alpha=0.2)



plt.title('Without Outlier')
plt.ylim(0,0.22)
plt.xlim(0,len(EW))
plt.legend()
plt.ylabel('D1')
plt.xlabel('Number of Inspection')
plt.show()

"""
#.................................OC-SVM----------------------------------------------------#

# OCSVM hyperparameters
# OCSVM hyperparameters
nu=0.08
EW_Train=EW[0:len(Out_Ewma_tr)]
EW_Vali=EW[len(Out_Ewma_tr):len(Out_Ewma_tr)+len(Out_Ewma_Vali)]
EW_D1=EW[len(Out_Ewma_tr)+len(Out_Ewma_Vali):len(Out_Ewma_D1)+len(Out_Ewma_tr)+len(Out_Ewma_Vali)]
EW_D2=EW[len(Out_Ewma_D1)+len(Out_Ewma_tr)+len(Out_Ewma_Vali):len(Out_Ewma_D1)+len(Out_Ewma_tr)+len(Out_Ewma_Vali)+len(Out_Ewma_D2)]


KL_Train_reshape=np.array(EW_Train).reshape(-1,1)
KL_Vali_reshape=np.array(EW_Vali).reshape(-1,1)
KL_D1_reshape=np.array(EW_D1).reshape(-1,1)
KL_D2_reshape=np.array(EW_D2).reshape(-1,1)
gamma=1/(KL_Train_reshape.var())
# gamma=5
# Fit the One-Class SVM
# clf = OneClassSVM(gamma=gamma,kernel="rbf", nu=nu,tol=2e-2)
# clf.fit(KL_Train_reshape)
# y_pred_train = clf.predict(KL_Train_reshape)
# y_pred_test = clf.predict(KL_Vali_reshape)
# y_pred_outliers = clf.predict(KL_D1_reshape)
# y_error_train = y_pred_train[y_pred_train == -1].size
# y_error_test = y_pred_test[y_pred_test == -1].size
# y_error_outliers = y_pred_outliers[y_pred_outliers == -1].size    

# print(y_error_train,y_error_test, y_error_outliers)
arr_hea=np.full(KL_Train_reshape.size, 1)
arr_vali=np.full(KL_Vali_reshape.size, 1)
arr_D1=np.full(KL_D1_reshape.size, -1)
arr_D2=np.full(KL_D2_reshape.size, -1)

True_value= np.concatenate((arr_hea,arr_vali,arr_D1,arr_D2))

# True_value.append(arr_hea)
# True_value.append(arr_vali)
# True_value.append(arr_D1)
# True_value.append(arr_D2)



transform = Nystroem(gamma=gamma,kernel="rbf")
clf_sgd = SGDOneClassSVM(
    nu=nu,max_iter=2000,learning_rate='adaptive',eta0=1e-3, shuffle=True, fit_intercept=True, tol=1e-7)
pipe_sgd = make_pipeline(transform, clf_sgd)
pipe_sgd.fit(KL_Train_reshape)
y_pred_train_sgd = pipe_sgd.predict(KL_Train_reshape)
y_pred_test_sgd = pipe_sgd.predict(KL_Vali_reshape)
y_pred_outliers_sgd = pipe_sgd.predict(KL_D1_reshape)
y_pred_outliers2_sgd = pipe_sgd.predict(KL_D2_reshape)

n_error_train_sgd = y_pred_train_sgd[y_pred_train_sgd == -1].size
n_error_test_sgd = y_pred_test_sgd[y_pred_test_sgd == -1].size
n_error_pos_outliers_sgd = y_pred_outliers_sgd[y_pred_outliers_sgd == 1].size
n_error_pos_outliers2_sgd = y_pred_outliers2_sgd[y_pred_outliers2_sgd == 1].size
n_error_neg_outliers_sgd = y_pred_outliers_sgd[y_pred_outliers_sgd == -1].size
n_error_neg_outliers2_sgd = y_pred_outliers2_sgd[y_pred_outliers2_sgd == -1].size
print(n_error_train_sgd,n_error_test_sgd, n_error_pos_outliers_sgd,n_error_pos_outliers2_sgd,n_error_neg_outliers_sgd,n_error_neg_outliers2_sgd)
##

EW_Value=np.array(EW).reshape(-1,1)

EW_predict = pipe_sgd.predict(EW_Value)

n_error_EW_predict = EW_predict[EW_predict == -1].size
n_error_EW_predict_neg = EW_predict[EW_predict == 1].size

print(n_error_EW_predict, n_error_EW_predict_neg)

x = np.arange(len(EW_predict))
p1=ax2.plot(x,EW_predict,linestyle='--',c ="k", lw=.05)
ax2.fill_between(x, -1, EW_predict, where=EW_predict>=1, facecolor='y', interpolate=True,alpha=0.1)
ax2.fill_between(x, 1, EW_predict, where=EW_predict<=0, facecolor= 'b', interpolate=True,alpha=0.1)
p2 = ax2.fill(np.NaN, np.NaN, 'y', alpha=0.3)
p3 = ax2.fill(np.NaN, np.NaN, 'b', alpha=0.3)
x = np.arange(len(True_value))
ax2.plot(x,True_value,linestyle='--',c ="k", lw=0.5)
ax2.legend([(p2[0], p1[0]), (p3[0], p1[0]),], ['Healthy','Damage'])
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
          ncol=3, fancybox=False, shadow=False)
# ax2.fill_between(x, -1, True_value, where=True_value>=1, facecolor='blue', interpolate=True,alpha=0.1)
# ax2.fill_between(x, 1, True_value, where=True_value<=0, facecolor= 'yellow', interpolate=True,alpha=0.1)

ax1.set_xlabel('Number of Inspection')
ax1.set_ylabel('DI')
ax2.set_ylabel('SVM', color='k')
ax2.set_yticklabels([])
plt.tight_layout()