# PCA
An SVD implementation of Principal Component Analysis, and a scaled nonlinear version of PCA

Principal component analysis (PCA) is a dimensionality reduction technique, which transforms the data matrix X (the rows denote observations and columns denote variables) such that each transformed dimension explains the maximum variance in the original data set. Commonly, PCA has been implemented by computing the eigen values and eigen vectors of X'X, where ' is the transpose operator. This approach is more memory intensive due to the matrix multiplication and storage of the result in memory. It also introduces floating point errors arising from the multiplication, particularly for -1<X(i,j)<1, where X(i,j) denotes the element in the ith row and jth column.

In this project, I have implemented a Python code that computes the scores and loading matrices of X by computing its singular value decomposition (SVD). If U*S*V' = svd(X), then U is the set of orthonormal eigenvectors of XX', S is a diagonal matrix of the singular values, and V is the set of orthonormal eigenvectors of X'X.

# Note
The PCA implemented in this project can only be used for analysis of data consisting of continuous variables. It cannot be used for discrete variables or a mix of continuous and discrete variables.

# Software and Packages Used
The programming is carried out in Python. The following software and packages are required to execute this file.
1. Python
2. numpy
3. matplotlib

A test data set is provided to demonstrate the execution of this project.
