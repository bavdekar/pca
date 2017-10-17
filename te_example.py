"""
Example code to test the PCA function

Created by Vinay Bavdekar
See license on Github
Ver. 0: 16 October 2017

This example uses data from the famous Tennessee-Eastman (TE) problem
setup. The data has 1 normal operating data-set and fault data sets for 21
different fault scenarios. For each scenario, there is a training data sets
and a test data set. The training data is used to compute the PCA matrices,
while the test data is used to test the efficacy of PCA.

In this example code, we take 3 data sets: 1 for normal data, and 2 for
2 different fault conditions.

In this example PCA is used to classify the data in to its different conditions,
i.e. PCA should be able to classify data as normal or associate it with the
corresponding fault condition.
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pca

# Load the training data
# Normal TE data
# This particular data set is stored as no. of variables x no. of observations.
# Hence needs to be transposed. Rest all data sets are observations x variables.
x_norm = np.loadtxt("TE_process/d00.dat")
x_norm = x_norm.T

# In this example we choose Fault-1 and Fault-2.
# Load Fault-1 data
x_f1 = np.loadtxt("TE_process/d01.dat")

# Load Fault-2 data
x_f2 = np.loadtxt("TE_process/d02.dat")

# Concatenate the data
x_data = np.concatenate((x_norm,x_f1,x_f2))

# Perform regular or scaled PCA
# Get the loadings matrix
P = pca.lpca(x_data,cv=96,normalize=True)

# Test the performance of PCA on the test data
x_norm = np.loadtxt("TE_process/d00_te.dat")
x_f1 = np.loadtxt("TE_process/d01_te.dat")
x_f2 = np.loadtxt("TE_process/d02_te.dat")

# If using regular PCA center the test data around the mean and variance
x_norm_c = x_norm - np.matmul(np.ones((x_norm.shape[0],x_norm.shape[1])),np.diag(np.mean(np.concatenate((x_norm,x_f1,x_f2)),axis=0)))
x_f1_c   = x_f1 - np.matmul(np.ones((x_f1.shape[0],x_f1.shape[1])),np.diag(np.mean(np.concatenate((x_norm,x_f1,x_f2)),axis=0)))
x_f2_c   = x_f2 - np.matmul(np.ones((x_f2.shape[0],x_f2.shape[1])),np.diag(np.mean(np.concatenate((x_norm,x_f1,x_f2)),axis=0)))


# Compute the scores of the test data
T_norm = np.matmul(x_norm_c,P.T)
T_f1 = np.matmul(x_f1_c,P.T)
T_f2 = np.matmul(x_f2_c,P.T)
T_c = np.concatenate((T_norm,T_f1,T_f2))

fig = plt.figure(figsize=(7,5))
#ax = fig.add_subplot(111,projection='3d')
ax = Axes3D(fig)
ax.scatter(T_norm[:,0],T_norm[:,1],T_norm[:,2],s=50,c='r', marker='o',label='Normal')
ax.scatter(T_f1[:,0],T_f1[:,1],T_f1[:,2],s=50,c='b',marker='^', label='Fault-1')
ax.scatter(T_f2[:,0],T_f2[:,1],T_f2[:,2],s=50,c='g',marker='+', label='Fault-2')
ax.legend()
ax.azim = 300
plt.show()
