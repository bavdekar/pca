# PCA
An SVD implementation of Principal Component Analysis, and a scaled nonlinear version of PCA

Principal component analysis (PCA) is a dimensionality reduction technique, which transforms the data matrix X (the rows denote observations and columns denote variables) such that each transformed dimension explains the maximum variance in the original data set. Commonly, PCA has been implemented by computing the eigen values and eigen vectors of X'X, where ' is the transpose operator. This approach is more memory intensive due to the matrix multiplication and storage of the result in memory. It also introduces floating point errors arising from the multiplication, particularly for -1<X(i,j)<1, where X(i,j) denotes the element in the ith row and jth column.

In this project, I have implemented a Python code that computes the scores and loading matrices of X by computing its singular value decomposition (SVD). If U*S*V' = svd(X), then U is the set of orthonormal eigenvectors of XX', S is a diagonal matrix of the singular values, and V is the set of orthonormal eigenvectors of X'X.

# Note
The PCA implemented in this project can only be used for analysis of data consisting of continuous variables. It cannot be used for discrete variables or a mix of continuous and discrete variables.

# Software and Packages Used
The programming is carried out in Python. The following software and libraries are required to execute this file.
1. Python3
2. numpy
3. scipy
4. matplotlib
5. mpl_toolkits

# The Tennessee-Eastman Data Set
The Tennessee-Eastman data set is a benchmark data set that is used to test pattern classification, fault detection and diagnosis. This data set is used here to test the PCA algorithms, and a copy of the data has been uploaded to my github. All the data is stored in the folder TE_process. The data relates to observations recorded for 52 process variables. The folder contains 44 data files, of which 22 are for training purpose and the remaining 22 are for testing purpose. The file d00.dat represents the training data for normal operating conditions, while the file d00_te.dat represents the test data for normal operating conditions. The remaining files d*.dat and d*_te.dat represent the training data and test data sets for fault conditions 01-21.

This data set can be obtained from Prof. Richard Braatz's website, the link for which is:

http://web.mit.edu/braatzgroup/links.html

The data set has its own license of use and distribution, however free use and distribution is permitted. Please go through the readme file for the data set provided in the folder.
