# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:49:02 2022

@author: muhamzs
Plotting file for AE Paper 3:
Experimental Data_Set

"""


import matplotlib
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal

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


#*************************************Loading Variables************************************************#
S1_train=np.log(np.load('S1_train.npy'))
S2_train=np.log(np.load('S2_train.npy'))
S3_train=np.log(np.load('S3_train.npy'))

S1_Vali=np.log(np.load('S1_Vali.npy'))
S2_Vali=np.log(np.load('S2_Vali.npy'))
S3_Vali=np.log(np.load('S3_Vali.npy'))




S1_D1=np.log(np.load('S1_D1.npy'))
S2_D1=np.log(np.load('S2_D1.npy'))
S3_D1=np.log(np.load('S3_D1.npy'))


S1_D2=np.log(np.load('S1_D2.npy'))
S2_D2=np.log(np.load('S2_D2.npy'))
S3_D2=np.log(np.load('S3_D2.npy'))


S1_D3=np.log(np.load('S1_D3.npy'))
S2_D3=np.log(np.load('S2_D3.npy'))
S3_D3=np.log(np.load('S3_D3.npy'))

#***********************************Fitting Data for training*****************************************#
N=800
Fleet_size=60
S1_train_base=S1_train[0:N]
S1_train_test=S1_train[N:len(S1_train)]

S2_train_base=S2_train[0:N]
S2_train_test=S2_train[N:len(S2_train)]

S3_train_base=S3_train[0:N]
S3_train_test=S3_train[N:len(S3_train)]


#**************************************
mu_s1, sigma_s1 = scipy.stats.norm.fit(S1_train_base)
mu_s2, sigma_s2 = scipy.stats.norm.fit(S2_train_base)
mu_s3, sigma_s3 = scipy.stats.norm.fit(S3_train_base)



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


    mean_s1_tr.append(mu_s1_tr)
    sigma_s1_tr.append(si_s1_tr)
    mean_s2_tr.append(mu_s2_tr)
    sigma_s2_tr.append(si_s2_tr)
    mean_s3_tr.append(mu_s3_tr)
    sigma_s3_tr.append(si_s3_tr)

    
    
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

    mean_s1_val.append(mu_s1_val)
    sigma_s1_val.append(si_s1_val)
    mean_s2_val.append(mu_s2_val)
    sigma_s2_val.append(si_s2_val)
    mean_s3_val.append(mu_s3_val)
    sigma_s3_val.append(si_s3_val)

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

    mean_s1_d1.append(mu_s1_d1)
    sigma_s1_d1.append(si_s1_d1)
    mean_s2_d1.append(mu_s2_d1)
    sigma_s2_d1.append(si_s2_d1)
    mean_s3_d1.append(mu_s3_d1)
    sigma_s3_d1.append(si_s3_d1)

    
    
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

    mean_s1_d2.append(mu_s1_d2)
    sigma_s1_d2.append(si_s1_d2)
    mean_s2_d2.append(mu_s2_d2)
    sigma_s2_d2.append(si_s2_d2)
    mean_s3_d2.append(mu_s3_d2)
    sigma_s3_d2.append(si_s3_d2)

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

    mean_s1_d3.append(mu_s1_d3)
    sigma_s1_d3.append(si_s1_d3)
    mean_s2_d3.append(mu_s2_d3)
    sigma_s2_d3.append(si_s2_d3)
    mean_s3_d3.append(mu_s3_d3)
    sigma_s3_d3.append(si_s3_d3)


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
