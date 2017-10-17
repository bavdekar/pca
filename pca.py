"""
Code created by Vinay A. Bavdekar
See license on Github repository
Ver. 0: 15 October 2017
"""
import numpy as np
import scipy as sp

def lpca(data,cv=None,normalize=False):
    # This function computes the loadings matrix of the data using the regular
    # linear principal components analysis
    # The loadings matrix is computed using the SVD of the data

    # data = nxm numpy matrix, where n=no. of observations, and m=no. of variables
    # cv = default cumulative variance of PCs
    # normalize = Boolean variable (True or False)
    if normalize==True:
        #Normalize all data to a zero-mean unit-variance
        m = np.mean(data,axis=0) #mean of each variable
        v = np.var(data,axis=0,ddof=1) #variance of each variable
        data_zm_uv = (data - np.matmul(np.ones((data.shape[0],data.shape[1])),np.diag(m)))/(v.T)
    else:
        data_zm_uv = data

    # Computes the SVD of the normalized (if True) data
    U,S,Vt = np.linalg.svd(data_zm_uv)

    # Compute cumulative variance explained by singular values in S
    cve = np.cumsum(100*S/np.sum(S))

    # Compute the number of dimensions at which to truncate based on the
    # cumulative variance
    if cv==None:
        cv = 95
    nx = np.argmax(cve>=cv)

    # Only the loadings matrix is required for computing scores of new data and
    # the Hoteling T statistic
    # The loadings matrix is Vt[0:nx,:], which is returned back from the function
    return Vt[0:nx,:]

def scpa(data,cv=None,normalize=False):
    # This function returns the loading matrix for a nonlinear scaled version
    # of principal components analysis
    # Note: For scaled PCA, the data is not normalized, hence the "normalize"
    # variable is just a dummy kept to maintain uniformity across functions
    if normalize==True:
        normalize=False

    # Compute the Pearson's correlation coefficient (PCC)
    # The PCC is scaled by 0.1 to avoid numerical ill conditioning
    # This is a tuning parameter and can be modified based on the nature of the
    # data.
    W = sp.linalg.exp(np.corrcoef(data,rowvar=False)/0.1)
    # Compute the sum of the rows of W and create a diagonal matrix of that sum
    D = np.diag(np.sum(W,axis=0))

    # Compute the scaled data set
    scaled_data = np.matmul(sp.linalg.inv(sp.linalg.sqrtm(D)),np.matmul(W,sp.linalg.inv(sp.linalg.sqrtm(D))))

    # Compute the SVD of the scaled data
    U,S,Vt = np.linalg.svd(scaled_data)

    # Compute cumulative variance explained by singular values in S
    cve = np.cumsum(100*S/np.sum(S))
    if cv==None:
        # Set the default cumulative variance
        cv = 95

    nx = np.argmax(cve>=cv)


    # Return the loading matrix after unscaling it
    return np.matmul(sp.linalg.inv(sp.linalg.sqrtm(D)),Vt.T)[:,0:nx].T
