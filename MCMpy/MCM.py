# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 11:24:29 2018

@author: mayank
"""

import numpy as np
#import pandas as pd
#from time import time
from sklearn.model_selection import StratifiedKFold
#import os
#from sklearn.cluster import KMeans
from sklearn.utils import resample
from scipy.stats import mode
#from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from numpy.matlib import repmat
from sklearn.metrics.pairwise import linear_kernel,rbf_kernel,manhattan_distances,polynomial_kernel,sigmoid_kernel,cosine_similarity,laplacian_kernel,paired_euclidean_distances,pairwise_distances
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from sklearn.kernel_approximation import RBFSampler, Nystroem
from numpy.linalg import eigh
#%%
#from scipy.io import loadmat
#from sklearn.decomposition import IncrementalPCA
#from sklearn import mixture

class MCM:
    def __init__(self, C1 = 1.0, C2 = 1e-05, C3 =1.0, C4 =1.0, problem_type ='classification', algo_type ='MCM' ,kernel_type = 'rbf', gamma = 1e-05, epsilon = 0.1, 
                 feature_ratio = 1.0, sample_ratio = 1.0, feature_sel = 'random', n_ensembles = 1,
                 batch_sz = 128, iterMax1 = 1000, iterMax2 = 1, eta = 0.01, tol = 1e-08, update_type = 'adam', 
                 reg_type = 'l1', combine_type = 'concat', class_weighting = 'balanced', upsample1 = False,
                 PV_scheme = 'kmeans', n_components = 100, do_pca_in_selection = False ):
        self.C1 = C1 #hyperparameter 1 #loss function parameter
        self.C2 = C2 #hyperparameter 2 #when using L1 or L2 or ISTA penalty
        self.C3 = C3 #hyperparameter 2 #when using elastic net penalty (this parameter should be between 0 and 1) or margin penalty value need not be between 0 and 1
        self.C4 = C4 #hyperparameter for final regressor or classifier used to ensemble when concatenating 
#        the outputs of previos layer of classifier or regressors
        self.problem_type = problem_type #{0:'classification', 1:'regression'}
        self.algo_type = algo_type #{0:MCM,1:'LSMCM'}
        self.kernel_type = kernel_type #{0:'linear', 1:'rbf', 2:'sin', 3:'tanh', 4:'TL1', 5:'linear_primal', 6:'rff_primal', 7:'nystrom_primal'}
        self.gamma = gamma #hyperparameter3 (kernel parameter for non-linear classification or regression)
        self.epsilon = epsilon #hyperparameter4 ( It specifies the epsilon-tube within which 
        #no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value.)
        self.n_ensembles = n_ensembles  #number of ensembles to be learnt, if setting n_ensembles > 1 then keep the sample ratio to be around 0.7
        self.feature_ratio = feature_ratio #percentage of features to select for each PLM
        self.sample_ratio = sample_ratio #percentage of data to be selected for each PLM
        self.batch_sz = batch_sz #batch_size
        self.iterMax1 = iterMax1 #max number of iterations for inner SGD loop
        self.iterMax2 = iterMax2 #max number of iterations for outer SGD loop
        self.eta = eta #initial learning rate
        self.tol = tol #tolerance to cut off SGD
        self.update_type = update_type #{0:'sgd',1:'momentum',3:'nesterov',4:'rmsprop',5:'adagrad',6:'adam'}
        self.reg_type = reg_type #{0:'l1', 1:'l2', 2:'en', 4:'ISTA', 5:'M'}#ISTA: iterative soft thresholding (proximal gradient), M: margin + l1
        self.feature_sel = feature_sel #{0:'sliding', 1:'random'}
        self.class_weighting = class_weighting #{0:'average', 1:'balanced'}
        self.combine_type = combine_type #{0:'concat',1:'average',2:'mode'}
        self.upsample1 = upsample1 #{0:False, 1:True}
        self.PV_scheme = PV_scheme # {0:'kmeans',1:'renyi'}
        self.n_components = n_components #number of components to choose as Prototype Vector set, or the number of features to form for kernel_approximation as in RFF and Nystroem 
        self.do_pca_in_selection = do_pca_in_selection #{0:False, 1:True}
        
    def add_bias(self,xTrain):
        N = xTrain.shape[0]
        if(xTrain.size!=0):
            xTrain=np.hstack((xTrain,np.ones((N,1))))
        return xTrain
    
    def standardize(self,xTrain):
        me=np.mean(xTrain,axis=0)
        std_dev=np.std(xTrain,axis=0)
        #remove columns with zero std
        idx=(std_dev!=0.0)
#        print(idx.shape)
        xTrain[:,idx]=(xTrain[:,idx]-me[idx])/std_dev[idx]
        return xTrain,me,std_dev
    
    def generate_samples(self,X_orig,old_imbalance_ratio,new_imbalance_ratio):
    
        N=X_orig.shape[0]
        M=X_orig.shape[1]
        neighbors_thresh=10
        new_samples=int(new_imbalance_ratio/old_imbalance_ratio*N - N)       
        #each point must generate these many samples 
        new_samples_per_point_orig=new_imbalance_ratio/old_imbalance_ratio - 1
        new_samples_per_point=int(new_imbalance_ratio/old_imbalance_ratio - 1)
        #check if the number of samples each point has to generate is > 1
        X1=np.zeros((0,M))   
            
        if(new_samples_per_point_orig>0 and new_samples_per_point_orig<=1):
            idx_samples=resample(np.arange(0,N), n_samples=int(N*new_samples_per_point_orig), random_state=1,replace=False)
            X=X_orig[idx_samples,]
            new_samples_per_point=1
            N=X.shape[0]
        else:
            X=X_orig
            
        if(N==1):
            X1=repmat(X,new_samples,1)            
        elif(N>1):        
            if(N<=neighbors_thresh):
                n_neighbors=int(N/2)
            else:
                n_neighbors=neighbors_thresh
                        
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X)                
            for i in range(N):
                #for each point find its n_neighbors nearest neighbors
                inds=nbrs.kneighbors(X[i,:].reshape(1,-1), n_neighbors, return_distance=False)
                temp_data=X[inds[0],:]    
                std=np.std(temp_data,axis=0)
                me=np.mean(temp_data,axis=0)
                np.random.seed(i)                
                x_temp=me + std*np.random.randn(new_samples_per_point,M)  
                X1=np.append(X1,x_temp,axis=0)
            
        return X_orig, X1      
    
    def upsample(self,X,Y,new_imbalance_ratio,upsample_type): 
        #xTrain: samples X features
        #yTrain : samples,
        #for classification only
        numClasses=np.unique(Y).size
        class_samples=np.zeros((numClasses,))
        X3=np.zeros((0,X.shape[1]))
        Y3=np.zeros((0,)) 
            
        #first find the samples per class per class
        for i in range(numClasses):
            idx1=(Y==i)
            class_samples[i]=np.sum(idx1)
            
        max_samples=np.max(class_samples)
    #    new_imbalance_ratio=0.5  
        if(upsample_type==1):
            old_imbalance_ratio_thresh=0.5
        else:
            old_imbalance_ratio_thresh=1
            
        for i in range(numClasses):
            idx1=(Y==i)
            old_imbalance_ratio=class_samples[i]/max_samples
            X1=X[idx1,:]
            Y1=Y[idx1,]              
    
            if(idx1.size==1):
                X1=np.reshape(X1,(1,X.shape[1]))
                
            if(old_imbalance_ratio<=old_imbalance_ratio_thresh and class_samples[i]!=0):               
                X1,X2=self.generate_samples(X1,old_imbalance_ratio,new_imbalance_ratio)
                new_samples=X2.shape[0]
                Y2=np.ones((new_samples,))
                Y2=Y2*Y1[0,]
                    
                #append original and generated samples
                X3=np.append(X3,X1,axis=0)
                X3=np.append(X3,X2,axis=0)
                
                Y3=np.append(Y3,Y1,axis=0)
                Y3=np.append(Y3,Y2,axis=0)            
            else:
                #append original samples only
                X3=np.append(X3,X1,axis=0)
                Y3=np.append(Y3,Y1,axis=0)
                
        Y3=np.array(Y3,dtype=np.int32)  
        return X3,Y3
    
    def kmeans_select(self,X,represent_points):
        """
        Takes in data and number of prototype vectors and returns the indices of the prototype vectors.
        The prototype vectors are selected based on the farthest distance from the kmeans centers
        Parameters
        ----------
        X: np.ndarray
            shape = n_samples, n_features
        represent_points: int
            number of prototype vectors to return
        do_pca: boolean
            whether to perform incremental pca for dimensionality reduction before selecting prototype vectors
            
        Returns
        -------
        sv: list
            list of the prototype vector indices from the data array given by X
        """
        do_pca = self.do_pca_in_selection
        N = X.shape[0]
        if(do_pca == True):
            if(X.shape[1]>50):
                n_components = 50
                ipca = IncrementalPCA(n_components=n_components, batch_size=np.min([128,X.shape[0]]))
                X = ipca.fit_transform(X)
    
        kmeans = MiniBatchKMeans(n_clusters=represent_points, batch_size=np.min([128,X.shape[0]]),random_state=0).fit(X)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        
        sv= []
        unique_labels = np.unique(labels).size 
        all_ind = np.arange(N)
        for j in range(unique_labels):
            X1 = X[labels == j,:]
            all_ind_temp = all_ind[labels==j]
            tempK = pairwise_distances(X1,np.reshape(centers[j,:],(1,X1.shape[1])))**2
            inds = np.argmax(tempK,axis=0)
            sv.append(all_ind_temp[inds[0]])
    
        return sv
    def renyi_select(self,X,represent_points):
        """
        Takes in data and number of prototype vectors and returns the indices of the prototype vectors.
        The prototype vectors are selected based on maximization of quadratic renyi entropy, which can be 
        written in terms of log sum exp which is a tightly bounded by max operator. Now for rbf kernel,
        the max_{ij}(-\|x_i-x_j\|^2) is equivalent to min_{ij}(\|x_i-x_j\|^2).
        Parameters
        ----------
        X: np.ndarray
            shape = n_samples, n_features
        represent_points: int
            number of prototype vectors to return
        do_pca: boolean
            whether to perform incremental pca for dimensionality reduction before selecting prototype vectors
            
        Returns
        -------
        sv: list
            list of the prototype vector indices from the data array given by X
        """
        do_pca = self.do_pca_in_selection
        N= X.shape[0]    
        capacity=represent_points
        selectionset=set([])
        set_full=set(list(range(N)))
        np.random.seed(1)
        if(len(selectionset)==0):
            selectionset = np.random.permutation(N)
            sv = list(selectionset)[0:capacity]        
        else:
            extrainputs = represent_points - len(selectionset)
            leftindices =list(set_full.difference(selectionset))
            info = np.random.permutation(len(leftindices))
            info = info[1:extrainputs]
            sv = selectionset.append(leftindices[info])
    
        if(do_pca == True):
            if(X.shape[1]>50): #takes more time
                n_components = 50
                ipca = IncrementalPCA(n_components=n_components, batch_size=np.min([128,X.shape[0]]))
                X = ipca.fit_transform(X)
            
        svX = X[sv,:]
        
        min_info = np.zeros((capacity,2))

        KsV = pairwise_distances(svX,svX)**2 #this is fast
        
        KsV[KsV==0] = np.inf
        min_info[:,1] = np.min(KsV,axis=1)
        min_info[:,0] = np.arange(capacity)
        minimum = np.min(min_info[:,1])
        counter = 0
        
        for i in range(N):
        #    find for which data the value is minimum
            replace = np.argmin(min_info[:,1])
            ids = int(min_info[min_info[:,0]==replace,0])
            #Subtract from totalcrit once for row 
            tempminimum = minimum - min_info[ids,1] 
            #Try to evaluate kernel function 
            
            tempsvX = np.zeros(svX.shape)
            tempsvX[:] = svX[:]
            inputX = X[i,:]
            tempsvX[replace,:] = inputX 
            tempK = pairwise_distances(tempsvX,np.reshape(inputX,(1,X.shape[1])))**2 #this is fast
            tempK[tempK==0] = np.inf
            distance_eval = np.min(tempK)
            tempminimum = tempminimum + distance_eval 
            if (minimum < tempminimum):
                minimum = tempminimum
                min_info[ids,1] = distance_eval
                svX[:] = tempsvX[:]
                sv[ids] = i
                counter +=1
        return sv
    
    def subset_selection(self,X,Y):
        n_components = self.n_components
        PV_scheme = self.PV_scheme
        problem_type = self.problem_type
        N = X.shape[0]
#        M = X.shape[1]
        numClasses = np.unique(Y).size
        
        use_global_sig = False
        use_global_sig1 = False
        if(use_global_sig ==True or problem_type == 'regression'):   
            if(PV_scheme == 'renyi'):
#                sig_global = np.power((np.std(X)*(np.power(N,(-1/(M+4))))),2) 
                subset = self.renyi_select(X,n_components)
            elif(PV_scheme == 'kmeans'):
                subset = self.kmeans_select(X,n_components)
            else:
                print('No PV_scheme provided... using all the samples!')
                subset = list(np.arange(N))
        else:
            all_samples = np.arange(N)
            subset=[]
            subset_per_class = np.zeros((numClasses,))
            class_dist = np.zeros((numClasses,))
            for i in range(numClasses):
                class_dist[i] = np.sum(Y == i)
                subset_per_class[i] = int(np.ceil((class_dist[i]/N)*n_components))
                
            for i in range(numClasses):
                xTrain = X[Y == i,]
                samples_in_class = all_samples[Y == i]
                N1 = xTrain.shape[0]
#                sig = np.power((np.std(xTrain)*(np.power(N1,(-1/(M+4))))),2)
                if(PV_scheme == 'renyi'):
                    if(use_global_sig1 == False):
                        subset1 = self.renyi_select(xTrain,int(subset_per_class[i]))
                    else:
#                        sig_global = np.power((np.std(X)*(np.power(N,(-1/(M+4))))),2) 
                        subset1 = self.renyi_select(xTrain,int(subset_per_class[i]))
                elif(PV_scheme == 'kmeans'):
                    subset1 = self.kmeans_select(xTrain,int(subset_per_class[i]))
                else:
                    print('No PV_scheme provided... using all the samples!')
                    subset1 = list(np.arange(N1))
                temp=list(samples_in_class[subset1])
                subset.extend(temp)
                
        return subset
    
    def divide_into_batches_stratified(self,yTrain):
        batch_sz=self.batch_sz
        #data should be of the form samples X features
        N=yTrain.shape[0]    
        num_batches=int(np.ceil(N/batch_sz))
        sample_weights=list()
        numClasses=np.unique(yTrain).size
        idx_batches=list()
    
        skf=StratifiedKFold(n_splits=num_batches, random_state=1, shuffle=True)
        j=0
        for train_index, test_index in skf.split(np.zeros(N), yTrain):
            idx_batches.append(test_index)
            class_weights=np.zeros((numClasses,))
            sample_weights1=np.zeros((test_index.shape[0],))
            temp=yTrain[test_index,]
            for i in range(numClasses):
                idx1=(temp==i)
                class_weights[i]=1.0/(np.sum(idx1)+1e-09)#/idx.shape[0]
                sample_weights1[idx1]=class_weights[i]            
            sample_weights.append(sample_weights1)

            j+=1
        return idx_batches,sample_weights,num_batches
    def kernel_transform(self, X1, X2 = None, kernel_type = 'linear_primal', n_components = 100, gamma = 1.0):
        """
        X1: n_samples1 X M
        X2: n_samples2 X M
        X: n_samples1 X n_samples2 : if kernel_type is non primal
        X: n_samples1 X n_components : if kernel_type is primal
        """
        if(kernel_type == 'linear'):
            X = linear_kernel(X1,X2)
        elif(kernel_type == 'rbf'):
            X = rbf_kernel(X1,X2,1/(2*gamma))   
        elif(kernel_type == 'tanh'):
            X = sigmoid_kernel(X1,X2,-gamma) 
        elif(kernel_type == 'sin'):
            X = np.sin(gamma*manhattan_distances(X1,X2))
        elif(kernel_type =='TL1'):                
            X = np.maximum(0,gamma - manhattan_distances(X1,X2)) 
        elif(kernel_type == 'rff_primal'):
            rbf_feature = RBFSampler(gamma=gamma, random_state=1, n_components = n_components)
            X = rbf_feature.fit_transform(X1)
        elif(kernel_type == 'nystrom_primal'):
            #cannot have n_components more than n_samples1
            if(n_components > X1.shape[0]):
                n_components  = X1.shape[0]
                self.n_components = n_components
            rbf_feature = Nystroem(gamma=gamma, random_state=1, n_components = n_components)
            X = rbf_feature.fit_transform(X1)
        elif(kernel_type == 'linear_primal'):                
            X = X1
        else:
            print('No kernel_type passed: using linear primal solver')
            X = X1
        return X
    
    def margin_kernel(self, X1, kernel_type = 'linear', gamma =1.0):
        """
        X1: n_samples1 X M
        X: n_samples1 X n_samples1 : if kernel_type is non primal
        """
        
        if(kernel_type == 'linear'):
            X = linear_kernel(X1,X1)
        elif(kernel_type == 'rbf'):
            X = rbf_kernel(X1,X1,1/(2*gamma))   
        elif(kernel_type == 'tanh'):
            X = sigmoid_kernel(X1,X1,-gamma) 
        elif(kernel_type == 'sin'):
            X = np.sin(gamma*manhattan_distances(X1,X1))
        elif(kernel_type =='TL1'):                
            X = np.maximum(0,gamma - manhattan_distances(X1,X1)) 
        else:
            print('no kernel_type, returning None')
            return None
        return X
    
    def matrix_decomposition(self, X):
        """
        Finds the matrices consisting of positive and negative parts of kernel matrix X
        Parameters:
        ----------
        X: n_samples X n_samples

        Returns:
        --------
        K_plus: kernel corresponding to +ve part
        K_minus: kernel corresponding to -ve part            
        """
        [D,U]=eigh(X)
        U_plus = U[:,D>0.0]
        U_minus = U[:,D<=0.0]
        D_plus = np.diag(D[D>0.0])
        D_minus = np.diag(D[D<=0.0])
        K_plus = np.dot(np.dot(U_plus,D_plus),U_plus.T)
        K_minus = -np.dot(np.dot(U_minus,D_minus),U_minus.T)
        return K_plus, K_minus
    
    def inner_opt(self, X, Y, data1, level):
        gamma = self.gamma
        kernel_type = self.kernel_type
        iterMax2 = self.iterMax2
        iterMax1 = self.iterMax1
        tol = self.tol
        algo_type = self.algo_type
        #if data1 = None implies there is no kernel computation, i.e., there is only primal solvers applicable
        if(data1 is not None):
            if(self.reg_type == 'M'):                
                K = self.margin_kernel( X1 = data1, kernel_type = kernel_type, gamma = gamma)
                if(kernel_type == 'linear' or kernel_type =='rbf' or kernel_type =='sin' or kernel_type =='tanh' or kernel_type =='TL1'):
                    K_plus, K_minus = self.matrix_decomposition(K)
                    
                    if(algo_type == 'MCM'):
                        W_prev,f,iters,fvals = self.train(X, Y, level, K_plus = K_plus, K_minus = None, W = None) 
                    elif(algo_type == 'LSMCM'):
                        W_prev,f,iters,fvals = self.train_LSMCM(X, Y, level, K_plus = K_plus, K_minus = None, W = None) 
                    else:
                        print('Wrong algo selected! Using MCM instead!')
                        W_prev,f,iters,fvals = self.train(X, Y, level, K_plus = K_plus, K_minus = None, W = None) 
                        
                    if(kernel_type == 'linear' or kernel_type == 'rbf'):
                        #for mercer kernels no need to train for outer loop
                        print('Returning for mercer kernels')
                        return W_prev,f,iters,fvals
                    else:
                        print('Solving for non - mercer kernels')
                        #for non mercer kernels, train for outer loop with initial point as W_prev
                        W_best = np.zeros(W_prev.shape)
                        W_best[:] = W_prev[:]
                        f_best = np.inf
                        iter_best = 0
                        fvals = np.zeros((iterMax1+1,))
                        iters = 0
                        fvals[iters] = f
                        rel_error = 1.0
                        print('iters =%d, f_outer = %0.9f'%(iters,f))
                        while(iters < iterMax2 and rel_error > tol):
                            iters = iters + 1 
                            
                            if(algo_type == 'MCM'):
                                W,f,iters1,fvals1 = self.train(X, Y, level, K_plus = K_plus, K_minus = None, W = W_prev) 
                            elif(algo_type == 'LSMCM'):
                                W,f,iters1,fvals1 = self.train_LSMCM(X, Y, level, K_plus = K_plus, K_minus = None, W = W_prev) 
                            else:
                                print('Wrong algo selected! Using MCM instead!')
                                W,f,iters1,fvals1 = self.train(X, Y, level, K_plus = K_plus, K_minus = None, W = W_prev)                         
                            
                            rel_error = np.abs((np.linalg.norm(W,'fro')-np.linalg.norm(W_prev,'fro'))/(np.linalg.norm(W_prev,'fro') + 1e-08))
                            W_prev[:] = W[:]
                            print('iters =%d, f_outer = %0.9f'%(iters,f))
                            if(f < f_best):
                                W_best[:] = W[:]
                                f_best = f
                                iter_best = iters
                            else:
                                break
                        fvals[iters] = -1
                        return W_best,f_best,iter_best,fvals
                else:
                    print('Please choose a kernel_type from linear, rbf, sin, tanh or TL1 for reg_type = M to work ')
                    print('Using a linear kernel')
                    self.kernel_type = 'linear'
                    K_plus, K_minus = self.matrix_decomposition(K)
                    
                    if(algo_type == 'MCM'):
                        W_prev,f,iters,fvals = self.train(X, Y, level, K_plus = K_plus, K_minus = None, W = None)  
                    elif(algo_type == 'LSMCM'):
                        W_prev,f,iters,fvals = self.train_LSMCM(X, Y, level, K_plus = K_plus, K_minus = None, W = None)  
                    else:
                        print('Wrong algo selected! Using MCM instead!')
                        W_prev,f,iters,fvals = self.train(X, Y, level, K_plus = K_plus, K_minus = None, W = None) 
                        
                    return W_prev,f,iters,fvals
            else:
                #i.e., reg_type is not M, then train accordingly using either l1, l2, ISTA or elastic net penalty
                if(algo_type == 'MCM'):
                    W,f,iters,fvals = self.train(X, Y, level, K_plus = None, K_minus = None, W = None)
                elif(algo_type == 'LSMCM'):
                    W,f,iters,fvals = self.train_LSMCM(X, Y, level, K_plus = None, K_minus = None, W = None)
                else:
                    print('Wrong algo selected! Using MCM instead!')
                    W,f,iters,fvals = self.train(X, Y, level, K_plus = None, K_minus = None, W = None)
                return W, f, iters, fvals                
        else:
            #i.e., data1 is None -> we are using primal solvers with either l1, l2, ISTA or elastic net penalty
            if(self.reg_type == 'M'): 
                print('Please choose a kernel_type from linear, rbf, sin, tanh or TL1 for reg_type = M to work')
                print('doing linear classifier with l1 norm on weights')
                self.reg_type = 'l1'
                self.C3 = 0.0
                
                if(algo_type == 'MCM'):
                    W,f,iters,fvals = self.train(X,Y,level, K_plus = None, K_minus = None, W = None)
                elif(algo_type == 'LSMCM'):
                    W,f,iters,fvals = self.train_LSMCM(X,Y,level, K_plus = None, K_minus = None, W = None)
                else:
                    print('Wrong algo selected! Using MCM instead!')
                    W,f,iters,fvals = self.train(X,Y,level, K_plus = None, K_minus = None, W = None)
                    
                return W,f,iters,fvals
            else:
                if(algo_type == 'MCM'):
                    W,f,iters,fvals = self.train(X,Y,level, K_plus = None, K_minus = None, W = None)
                elif(algo_type == 'LSMCM'):
                    W,f,iters,fvals = self.train_LSMCM(X,Y,level, K_plus = None, K_minus = None, W = None)
                else:
                    print('Wrong algo selected! Using MCM instead!')
                    W,f,iters,fvals = self.train(X,Y,level, K_plus = None, K_minus = None, W = None)

                return W,f,iters,fvals           
        
        return W,f,iters,fvals
        
    def select_(self, xTest, xTrain, kernel_type, subset, idx_features, idx_samples):
        #xTest corresponds to X1
        #xTrain corresponds to X2 
        if(kernel_type == 'linear' or kernel_type =='rbf' or kernel_type =='sin' or kernel_type =='tanh' or kernel_type =='TL1'):            
            X2 = xTrain[idx_samples,:]
            X2 = X2[:,idx_features] 
            X2 = X2[subset,]
            X1 = xTest[:,idx_features]
        else:
            X1 = xTest[:,idx_features]
            X2 = None
        return X1, X2
    
    def normalize_(self,xTrain, me, std):
        idx = (std!=0.0)
        xTrain[:,idx] = (xTrain[:,idx]-me[idx])/std[idx]
        return xTrain
    
    def fit(self,xTrain,yTrain):
        #xTrain: samples Xfeatures
        #yTrain: samples
        #for classification: entries of yTrain should be between {0 to numClasses-1}
        #for regresison  : entries of yTrain should be real values
        N = xTrain.shape[0]
        M = xTrain.shape[1]
        if(self.problem_type =='classification'):
            numClasses=np.unique(yTrain).size
        if(self.problem_type =='regression'):
            if(yTrain.size == yTrain.shape[0]):
                yTrain = np.reshape(yTrain,(yTrain.shape[0],1))
            numClasses = yTrain.shape[1] #for multi target SVM, assuming all targets are independent to each other

        feature_indices=np.zeros((self.n_ensembles,int(M*self.feature_ratio)),dtype=np.int32)
        sample_indices=np.zeros((self.n_ensembles,int(N*self.sample_ratio)),dtype=np.int32)
        
        W_all={}
        me_all= {}
        std_all = {}
        subset_all = {}
        if(self.combine_type=='concat'):    
            P_all=np.zeros((N,self.n_ensembles*numClasses)) #to concatenate the classes
            
        level=0            
        gamma = self.gamma
        kernel_type = self.kernel_type
        n_components = self.n_components
        for i in range(self.n_ensembles):
            print('training PLM %d'%i)
            
            if(self.sample_ratio!=1.0):
                idx_samples=resample(np.arange(0,N), n_samples=int(N*self.sample_ratio), random_state=i,replace=False)
            else:
                idx_samples = np.arange(N)
            
            if(self.feature_ratio!=1.0):
                idx_features=resample(np.arange(0,M), n_samples=int(M*self.feature_ratio), random_state=i,replace=False)
            else:
                idx_features = np.arange(0,M)   
                
            feature_indices[i,:] = idx_features
            sample_indices[i,:] = idx_samples
            
            xTrain_temp = xTrain[idx_samples,:]
            xTrain_temp = xTrain_temp[:,idx_features] 
            
            yTrain1 = yTrain[idx_samples,]
            
            if(kernel_type == 'linear' or kernel_type =='rbf' or kernel_type =='sin' or kernel_type =='tanh' or kernel_type =='TL1'):
                subset = self.subset_selection(xTrain_temp,yTrain1)
                data1 = xTrain_temp[subset,]
                subset_all[i] = subset
            else:
                subset_all[i] = []
                data1 = None

            xTrain1 = self.kernel_transform( X1 = xTrain_temp, X2 = data1, kernel_type = kernel_type, n_components = n_components, gamma = gamma)
            
            #standardize the dataset
            xTrain1, me, std  = self.standardize(xTrain1)
            
            me_all[i] = me
            std_all[i] = std
            
            if(self.problem_type == 'regression'):
                epsilon = self.epsilon
                N1 = yTrain1.shape[0]
                W = np.zeros((xTrain1.shape[1]+2,numClasses*2)) #2 is added to incorporate the yTrain2 and bias term appended to xTrain1
                for j in range(numClasses):
                    yTrain3 = np.append(np.ones((N1,)), np.zeros((N1,)))
                    yTrain2 = np.append(yTrain1[:,j] + epsilon, yTrain1[:,j] - epsilon, axis = 0)
                    xTrain2 = np.append(xTrain1, xTrain1, axis = 0)
                    xTrain2 = np.append(xTrain2, np.reshape(yTrain2,(2*N1,1)), axis =1)
#                    Wa,f,iters,fvals=self.train(xTrain2,yTrain3,level)
                    Wa,f,iters,fvals = self.inner_opt(xTrain2, yTrain3, data1, level)
                    W[:,j:j+2] = Wa
                W_all[i]=W # W will be of the shape (M+2,), here numClasses = 1
                
            if(self.problem_type == 'classification'):
#                W,f,iters,fvals=self.train(xTrain1,yTrain1,level)            
                W,f,iters,fvals = self.inner_opt(xTrain1, yTrain1, data1, level)
                W_all[i]=W # W will be of the shape (M+2,numClasses)
                
        if(self.n_ensembles == 1 or self.combine_type != 'concat'):
            return W_all, sample_indices, feature_indices, me_all, std_all, subset_all
        else:
            if(self.combine_type=='concat'):
                level=1
                for i in range(self.n_ensembles):
                    
                    X1, X2 = self.select_(xTrain, xTrain, kernel_type, subset_all[i], feature_indices[i,:], sample_indices[i,:])
                    xTrain1 = self.kernel_transform( X1 = X1, X2 = X2, kernel_type = kernel_type, n_components = n_components, gamma = gamma)
                    xTrain1 = self.normalize_(xTrain1,me_all[i],std_all[i])

                    M = xTrain1.shape[1]
                    xTrain1=self.add_bias(xTrain1)    
                    W = W_all[i]      
                    
                    if(self.problem_type == 'regression'):
                        scores = np.zeros((xTrain1.shape[0],numClasses))
                        for j in range(numClasses):
                            W2 = W[:,j:j+2]
                            W1 = (W2[:,0] - W2[:,1])/2
                            scores1 = xTrain1[:,0:M].dot(W1[0:M,]) + np.dot(xTrain1[:,M], W1[M+1,])
                            scores1 = -1.0/(W1[M,] + 1e-08)*scores1
                            scores[:,j] = scores1
                            
                    if(self.problem_type == 'classification'): 
                        scores = xTrain1.dot(W)
                        
                    P_all[:,i*numClasses:numClasses+i*numClasses] = scores
                    
                #train another regressor or classifier on top
                if(self.problem_type == 'regression'):
                    epsilon = self.epsilon                    
                    P_all_1 = np.zeros((P_all.shape[0],self.n_ensembles))
                    W1 = np.zeros((P_all_1.shape[1]+2,numClasses*2))
                    for j in range(numClasses):
                        for k in range(self.n_ensembles):
                            P_all_1[:,k] = P_all[:,numClasses*k+j]
                        yTrain3 = np.append(np.ones((N,)), np.zeros((N,)))
                        yTrain2 = np.append(yTrain[:,j] + epsilon, yTrain[:,j] - epsilon, axis = 0)
                        P_all_2 = np.append(P_all_1, P_all_1, axis = 0)
                        P_all_2 = np.append(P_all_2, np.reshape(yTrain2,(2*N,1)), axis =1)                
#                        Wa,f,iters,fvals = self.train(P_all_2,yTrain3,level)  
                        Wa,f,iters,fvals = self.inner_opt(P_all_2, yTrain3, None, level)
                        W1[:,j:j+2] = Wa
                        
                if(self.problem_type == 'classification'): 
#                    W1,f1,iters1,fvals1 = self.train(P_all,yTrain,level)
                    W1,f,iters,fvals = self.inner_opt(P_all, yTrain, None, level)
                    
                W_all[self.n_ensembles] = W1
                return W_all, sample_indices, feature_indices, me_all, std_all, subset_all
                
    def train(self, xTrain, yTrain, level, K_plus = None, K_minus = None, W = None):
        #min D(E|w|_1 + (1-E)*0.5*|W|_2^2) + C*\sum_i\sum_(j)|f_j(i)| + \sum_i\sum_(j_\neq y_i)max(0,(1-f_y_i(i) + f_j(i)))
        #setting C = 0 gives us SVM
        # or when using margin term i.e., reg_type = 'M'
        #min D(E|w|_1) + (E)*0.5*\sum_j=1 to numClasses (w_j^T(K+ - K-)w_j) + C*\sum_i\sum_(j)|f_j(i)| + \sum_i\sum_(j_\neq y_i)max(0,(1-f_y_i(i) + f_j(i)))
        #setting C = 0 gives us SVM with margin term
        if(self.upsample1==True):
            xTrain,yTrain=self.upsample(xTrain,yTrain,new_imbalance_ratio=0.5,upsample_type=1)
            
        xTrain=self.add_bias(xTrain)
        
        M=xTrain.shape[1]
        N=xTrain.shape[0]
        numClasses=np.unique(yTrain).size
        verbose = False
        if(level==0):
            C = self.C1 #for loss function of MCM
            D = self.C2 #for L1 or L2 penalty
            E = self.C3 #for elastic net penalty or margin term
        else:
            C = self.C4 #for loss function of MCM 
            D = self.C2 #for L1 or L2 penalty
            E = self.C3 #for elastic net penalty since in combining the classifiers we use a linear primal classifier
            
        iterMax1 = self.iterMax1
        eta_zero = self.eta
        class_weighting = self.class_weighting
        reg_type = self.reg_type
        update_type = self.update_type
        tol = self.tol
        np.random.seed(1)
        
        if(W is None):
            W=0.001*np.random.randn(M,numClasses)
            W=W/np.max(np.abs(W))
        else:
            W_orig = np.zeros(W.shape)
            W_orig[:] = W[:]
        
        class_weights=np.zeros((numClasses,))
        sample_weights=np.zeros((N,))
        #divide the data into K clusters
    
        for i in range(numClasses):
            idx=(yTrain==i)           
            class_weights[i]=1.0/np.sum(idx)
            sample_weights[idx]=class_weights[i]
                        
        G_clip_threshold = 100
        W_clip_threshold = 500
        eta=eta_zero
                       
        scores = xTrain.dot(W) #samples X numClasses
        N = scores.shape[0]
        correct_scores = scores[range(N),np.array(yTrain,dtype='int32')]
        mat = (scores.transpose()-correct_scores.transpose()).transpose() 
        mat = mat+1.0
        mat[range(N),np.array(yTrain,dtype='int32')] = 0.0
        thresh1 = np.zeros(mat.shape)
        thresh1[mat>0.0] = mat[mat>0.0] #for the SVM loss 
        
        f=0.0
        if(reg_type=='l2'):
            f += D*0.5*np.sum(W**2) 
        if(reg_type=='l1'):
            f += D*np.sum(np.abs(W))
        if(reg_type=='en'):
            f += D*0.5*(1-E)*np.sum(W**2)  +  D*E*np.sum(np.abs(W))
            
            
        if(class_weighting=='average'):
            f1 = C*np.sum(np.abs(scores)) + np.sum(thresh1)
            f += (1.0/N)*f1 
        else:
            f1 = C*np.sum(np.abs(scores)*sample_weights[:,None]) + np.sum(thresh1*sample_weights[:,None])
            f+= (1.0/numClasses)*f1 
        
        if(K_minus is not None):
            temp_mat = np.dot(K_minus,W_orig[0:(M-1),])
        
        
        for i in range(numClasses):
            #add the term (E/2*numclasses)*lambda^T*K_plus*lambda for margin
            if(K_plus is not None):
                w = W[0:(M-1),i]
                f2 = np.dot(np.dot(K_plus,w),w)
                f+= ((0.5*E)/(numClasses))*f2  
             #the second term in the objective function
            if(K_minus is not None):
                f3 = np.dot(temp_mat[:,i],w)
                f+= -((0.5*E)/(numClasses))*f3
        
        
        iter1=0
        print('iter1=%d, f=%0.3f'%(iter1,f))
                
        f_best=f
        fvals=np.zeros((iterMax1+1,))
        fvals[iter1]=f_best
        W_best=np.zeros(W.shape)
        iter_best=iter1
        f_prev=f_best
        rel_error=1.0
#        f_prev_10iter=f
        
        if(reg_type=='l1' or reg_type =='en' or reg_type == 'M'):
            # from paper: Stochastic Gradient Descent Training for L1-regularized Log-linear Models with Cumulative Penalty
            if(update_type == 'adam' or update_type == 'adagrad' or update_type == 'rmsprop'):
                u = np.zeros(W.shape)
            else:
                u = 0.0
            q=np.zeros(W.shape)
            z=np.zeros(W.shape)
            all_zeros=np.zeros(W.shape)
        
        eta1=eta_zero 
        v=np.zeros(W.shape)
        v_prev=np.zeros(W.shape)    
        vt=np.zeros(W.shape)
        m=np.zeros(W.shape)
        vt=np.zeros(W.shape)
        
        cache=np.zeros(W.shape)
        eps=1e-08
        decay_rate=0.99
        mu1=0.9
        mu=mu1
        beta1 = 0.9
        beta2 = 0.999  
        iter_eval=10 #evaluate after every 10 iterations
        
        idx_batches, sample_weights_batch, num_batches = self.divide_into_batches_stratified(yTrain)
        while(iter1<iterMax1 and rel_error>tol):
            iter1=iter1+1            
            for batch_num in range(0,num_batches):
    #                batch_size=batch_sizes[j]
                test_idx=idx_batches[batch_num]
                data=xTrain[test_idx,]
                labels=yTrain[test_idx,] 
                N=labels.shape[0]
                scores=data.dot(W)
                correct_scores=scores[range(N),np.array(labels,dtype='int32')]#label_batches[j] for this line should be in the range [0,numClasses-1]
                mat=(scores.transpose()-correct_scores.transpose()).transpose() 
                mat=mat+1.0
                mat[range(N),np.array(labels,dtype='int32')]=0.0
                
                thresh1=np.zeros(mat.shape)
                thresh1[mat>0.0]=mat[mat>0.0]
                
                binary1 = np.zeros(thresh1.shape)
                binary1[thresh1>0.0] = 1.0                
                
                row_sum=np.sum(binary1,axis=1)
                binary1[range(N),np.array(labels,dtype='int32')]=-row_sum
                
                
                if(C !=0.0):
                    binary2 = np.zeros(scores.shape)
                    binary2[scores>0.0] = 1.0                
                    binary2[scores<0.0] = -1.0
                else:
                    binary2 = 0
                    
                dscores1 = binary1
                dscores2 = binary2
                if(class_weighting=='average'):
                    gradW = np.dot((dscores1 + C*dscores2).transpose(),data)
                    gradW=gradW.transpose()
                    gradW = (1.0/N)*gradW
#                    gradW += gradW1 - gradW2
                else:
                    sample_weights_b=sample_weights_batch[batch_num]
                    gradW=np.dot((dscores1 + C*dscores2).transpose(),data*sample_weights_b[:,None])
                    gradW=gradW.transpose()
                    gradW=(1.0/numClasses)*gradW
#                    gradW += gradW1 - gradW2
                        
                if(np.sum(gradW**2)>G_clip_threshold):#gradient clipping
                    gradW = G_clip_threshold*gradW/np.sum(gradW**2)
                    
                if(update_type=='sgd'):
                    W = W - eta*gradW
                elif(update_type=='momentum'):
                    v = mu * v - eta * gradW # integrate velocity
                    W += v # integrate position
                elif(update_type=='nesterov'):
                    v_prev[:] = v[:] # back this up
                    v = mu * v - eta * gradW # velocity update stays the same
                    W += -mu * v_prev + (1 + mu) * v # position update changes form
                elif(update_type=='adagrad'):
                    cache += gradW**2
                    W += - eta1* gradW / (np.sqrt(cache) + eps)
                elif(update_type=='rmsprop'):
                    cache = decay_rate * cache + (1 - decay_rate) * gradW**2
                    W += - eta1 * gradW / (np.sqrt(cache) + eps)
                elif(update_type=='adam'):
                    m = beta1*m + (1-beta1)*gradW
                    mt = m / (1-beta1**(iter1+1))
                    v = beta2*v + (1-beta2)*(gradW**2)
                    vt = v / (1-beta2**(iter1+1))
                    W += - eta1 * mt / (np.sqrt(vt) + eps)           
                else:
                    W = W - eta*gradW
                    
                if(reg_type == 'M'):
                    gradW1= np.zeros(W.shape)
                    gradW2= np.zeros(W.shape)
                    for i in range(numClasses):
                        w=W[0:(M-1),i]
                        if(K_plus is not None):
                            gradW1[0:(M-1),i]=((E*0.5)/(numClasses))*2*np.dot(K_plus,w)
                        if(K_minus is not None):
                            gradW2[0:(M-1),i]=((E*0.5)/(numClasses))*temp_mat[:,i]
                    if(update_type == 'adam'):
                        W += -(gradW1-gradW2)*(eta1/(np.sqrt(vt) + eps)) 
                    elif(update_type == 'adagrad' or update_type =='rmsprop'):
                        W += -(gradW1-gradW2)*(eta1/(np.sqrt(cache) + eps))
                    else:
                        W += -(gradW1-gradW2)*(eta)
                        
                if(reg_type == 'ISTA'):
                    if(update_type == 'adam'):
                        idx_plus =  W > D*(eta1/(np.sqrt(vt) + eps))
                        idx_minus = W < -D*(eta1/(np.sqrt(vt) + eps))
                        idx_zero = np.abs(W) < D*(eta1/(np.sqrt(vt) + eps))
                        W[idx_plus] = W[idx_plus] - D*(eta1/(np.sqrt(vt[idx_plus]) + eps))
                        W[idx_minus] = W[idx_minus] + D*(eta1/(np.sqrt(vt[idx_minus]) + eps))
                        W[idx_zero] = 0.0
                    elif(update_type == 'adagrad' or update_type =='rmsprop'):
                        idx_plus =  W > D*(eta1/(np.sqrt(cache) + eps))
                        idx_minus = W < -D*(eta1/(np.sqrt(cache) + eps))
                        idx_zero = np.abs(W) < D*(eta1/(np.sqrt(cache) + eps))
                        W[idx_plus] = W[idx_plus] - D*(eta1/(np.sqrt(cache[idx_plus]) + eps))
                        W[idx_minus] = W[idx_minus] + D*(eta1/(np.sqrt(cache[idx_minus]) + eps))
                        W[idx_zero] = 0.0
                    else:
                        idx_plus =  W > D*(eta)
                        idx_minus = W < -D*(eta)
                        idx_zero = np.abs(W) < D*(eta)
                        W[idx_plus] = W[idx_plus] - D*(eta)
                        W[idx_minus] = W[idx_minus] + D*(eta)
                        W[idx_zero] = 0.0

                        
                if(reg_type=='l2'):
                    if(update_type == 'adam'):
                        W += -D*W*(eta1/(np.sqrt(vt) + eps)) 
                    elif(update_type == 'adagrad' or update_type =='rmsprop'):
                        W += -D*W*(eta1/(np.sqrt(cache) + eps))
                    else:
                        W += -D*W*(eta)  
                
                if(reg_type=='en'):
                    if(update_type == 'adam'):
                        W += -D*(1.0-E)*W*(eta1/(np.sqrt(vt) + eps)) 
                    elif(update_type == 'adagrad' or update_type =='rmsprop'):
                        W += -D*(1.0-E)*W*(eta1/(np.sqrt(cache) + eps))
                    else:
                        W += -D*W*(eta)  
                    
                if(reg_type=='l1' or reg_type == 'M'):
                    if(update_type=='adam'):
                        u = u + D*(eta1/(np.sqrt(vt) + eps))
                    elif(update_type == 'adagrad' or update_type =='rmsprop'):
                        u = u + D*(eta1/(np.sqrt(cache) + eps))
                    else:
                        u = u + D*eta
                    z[:] = W[:]
                    idx_plus = W>0
                    idx_minus = W<0
                    
                    W_temp = np.zeros(W.shape)
                    if(update_type=='adam' or update_type == 'adagrad' or update_type =='rmsprop'):
                        W_temp[idx_plus]=np.maximum(all_zeros[idx_plus],W[idx_plus]-(u[idx_plus]+q[idx_plus]))
                        W_temp[idx_minus]=np.minimum(all_zeros[idx_minus],W[idx_minus]+(u[idx_minus]-q[idx_minus]))                    
                    else:
                        W_temp[idx_plus]=np.maximum(all_zeros[idx_plus],W[idx_plus]-(u+q[idx_plus]))
                        W_temp[idx_minus]=np.minimum(all_zeros[idx_minus],W[idx_minus]+(u-q[idx_minus]))
                    
                    W[idx_plus]=W_temp[idx_plus]
                    W[idx_minus]=W_temp[idx_minus]
                    q=q+(W-z)
                    
                if(reg_type=='en'):
                    if(update_type=='adam'):
                        u = u + D*E*(eta1/(np.sqrt(vt) + eps))
                    elif(update_type == 'adagrad' or update_type =='rmsprop'):
                        u = u + D*E*(eta1/(np.sqrt(cache) + eps))                    
                    else:
                        u = u + D*E*eta
                    z[:] = W[:]
                    idx_plus = W>0
                    idx_minus = W<0
                    
                    W_temp = np.zeros(W.shape)
                    if(update_type=='adam' or update_type == 'adagrad' or update_type =='rmsprop'):
                        W_temp[idx_plus]=np.maximum(all_zeros[idx_plus],W[idx_plus]-(u[idx_plus]+q[idx_plus]))
                        W_temp[idx_minus]=np.minimum(all_zeros[idx_minus],W[idx_minus]+(u[idx_minus]-q[idx_minus]))                    
                    else:
                        W_temp[idx_plus]=np.maximum(all_zeros[idx_plus],W[idx_plus]-(u+q[idx_plus]))
                        W_temp[idx_minus]=np.minimum(all_zeros[idx_minus],W[idx_minus]+(u-q[idx_minus]))
                    W[idx_plus]=W_temp[idx_plus]
                    W[idx_minus]=W_temp[idx_minus]
                    q=q+(W-z)
                
                if(np.sum(W**2)>W_clip_threshold):#gradient clipping
                    W = W_clip_threshold*W/np.sum(W**2)
            
            if(iter1%iter_eval==0):                    
                #once the W are calculated for each epoch we calculate the scores
                scores=xTrain.dot(W)
#                scores=scores-np.max(scores)
                N=scores.shape[0]
                correct_scores = scores[range(N),np.array(yTrain,dtype='int32')]
                mat = (scores.transpose()-correct_scores.transpose()).transpose() 
                mat = mat+1.0
                mat[range(N),np.array(yTrain,dtype='int32')] = 0.0
                thresh1 = np.zeros(mat.shape)
                thresh1[mat>0.0] = mat[mat>0.0] #for the SVM loss 
                
                f=0.0
                if(reg_type=='l2'):
                    f += D*0.5*np.sum(W**2) 
                if(reg_type=='l1'):
                    f += D*np.sum(np.abs(W))
                if(reg_type=='en'):
                    f += D*0.5*(1-E)*np.sum(W**2)  +  D*E*np.sum(np.abs(W))
                    
                    
                if(class_weighting=='average'):
                    f1 = C*np.sum(np.abs(scores)) + np.sum(thresh1)
                    f += (1.0/N)*f1 
                else:
                    f1 = C*np.sum(np.abs(scores)*sample_weights[:,None]) + np.sum(thresh1*sample_weights[:,None])
                    f+= (1.0/numClasses)*f1 
                    
                for i in range(numClasses):
                    #first term in objective function for margin
                    if(K_plus is not None):
                        w = W[0:(M-1),i]
                        f2 = np.dot(np.dot(K_plus,w),w)
                        f += ((0.5*E)/(numClasses))*f2  
                        #the second term in the objective function for margin
                    if(K_minus is not None):
                        f3 = np.dot(temp_mat[:,i],w)
                        f += -((0.5*E)/(numClasses))*f3
                if(verbose == True):        
                    print('iter1=%d, f=%0.3f'%(iter1,f))
                fvals[iter1]=f
                rel_error=np.abs(f_prev-f)/np.abs(f_prev)
                max_W = np.max(np.abs(W))
                W[np.abs(W)<1e-03*max_W]=0.0
                if(f<f_best):
                    f_best=f
                    W_best[:]=W[:]
                    max_W = np.max(np.abs(W))
                    W_best[np.abs(W_best)<1e-03*max_W]=0.0
                    iter_best=iter1
                else:
                    break
                f_prev=f      
 
            eta=eta_zero/np.power((iter1+1),1)
            
        fvals[iter1]=-1
        return W_best,f_best,iter_best,fvals
            
 
    def predict(self,data, xTrain, W_all, sample_indices, feature_indices, me_all, std_all, subset_all):
        #type=2 -> mode of all labels
        #type=1 -> average of all labels
        #type=3 -> concat of all labels
        types = self.combine_type
        kernel_type = self.kernel_type
        gamma = self.gamma
        n_components = self.n_components
        
        n_ensembles = feature_indices.shape[0]
        N = data.shape[0]  
        M = data.shape[1]
        if(self.problem_type == 'classification'):
            numClasses = W_all[0].shape[1]
            label = np.zeros((N,))
        if(self.problem_type == 'regression'):
            numClasses =  int(W_all[0].shape[1]/2)
            print('numClasses=%d'%numClasses)
            label = np.zeros((N,numClasses))
#        print('numClasses =%d'%numClasses)
        
        if(types=='mode'):
            label_all_1 = np.zeros((N,n_ensembles))
            label_all_2 = np.zeros((N,n_ensembles*numClasses))
            for i in range(n_ensembles):
                
#                print('testing PLM %d'%i)
                X1, X2 = self.select_(data, xTrain, kernel_type, subset_all[i], feature_indices[i,:], sample_indices[i,:])
                data1 = self.kernel_transform(X1 = X1, X2 = X2, kernel_type = kernel_type, n_components = n_components, gamma = gamma)
                data1 = self.normalize_(data1,me_all[i],std_all[i])
                M = data1.shape[1]
                data1 = self.add_bias(data1)                    
                
                W = W_all[i]  
                
                if(self.problem_type == 'regression'):
                    scores = np.zeros((data1.shape[0],numClasses))
                    for j in range(numClasses):
                        W2 = W[:,j:j+2]
                        W1 = (W2[:,0] - W2[:,1])/2
                        scores1 = data1[:,0:M].dot(W1[0:M,]) + np.dot(data1[:,M], W1[M+1,])
                        scores1 = -1.0/(W1[M,] + 1e-08)*scores1
                        scores[:,j] = scores1
                    label_all_2[:,i*numClasses:i*numClasses+numClasses] = scores
                    
                if(self.problem_type == 'classification'):
                    scores = data1.dot(W)
                    label_all_1[:,i] = np.argmax(scores,axis=1) 
                    
            if(self.problem_type == 'classification'):
                label = mode(label_all_1,axis=1)[0]
                label = np.int32(np.reshape(label,(N,)))
                return label
                
            if(self.problem_type == 'regression'):
                label = np.zeros((N,numClasses))
                for j in range(numClasses):
                    label_temp = np.zeros((N,n_ensembles))
                    for k in range(n_ensembles):
                        label_temp[:,k] = label_all_2[:,k*numClasses+j]
                    label[:,j] = np.reshape(mode(label_temp,axis=1)[0],(label.shape[0],))
                return label                   
                        
                
        elif(types=='average'):
            label_all_2=np.zeros((N,numClasses))
            for i in range(n_ensembles):                
#                print('testing PLM %d'%i)
                X1, X2 = self.select_(data, xTrain, kernel_type, subset_all[i], feature_indices[i,:], sample_indices[i,:])
                data1 = self.kernel_transform( X1 = X1, X2 = X2, kernel_type = kernel_type, n_components = n_components, gamma = gamma)
                data1 = self.normalize_(data1,me_all[i],std_all[i])
                M = data1.shape[1]
                data1 = self.add_bias(data1)                                        
                
                W = W_all[i]  
                if(self.problem_type == 'regression'):
                    scores = np.zeros((data1.shape[0],numClasses))
                    for j in range(numClasses):
                        W2 = W[:,j:j+2]
                        W1 = (W2[:,0] - W2[:,1])/2
#                        W1 = (W[:,0]-W[:,1])/2
                        scores1 = data1[:,0:M].dot(W1[0:M,]) + np.dot(data1[:,M], W1[M+1,])
                        scores1 = -1.0/(W1[M,] + 1e-08)*scores1
                        scores[:,j] = scores1
                    label += label + scores/n_ensembles
                    
                if(self.problem_type == 'classification'):
                    scores = data1.dot(W)
                    label_all_2 += label_all_2 + scores
            
            if(self.problem_type == 'classification'):
                label=np.argmax(label_all_2,axis=1)
                return label
            if(self.problem_type == 'regression'):
                return label
                    
        elif(types =='concat'):
#            if(self.problem_type == 'regression'):
#                P_all=np.zeros((N,n_ensembles))
#            if(self.problem_type == 'classification'): 
            N = data.shape[0]               
            P_all=np.zeros((N,n_ensembles*numClasses))
                
            for i in range(n_ensembles):
#                print('testing PLM %d'%i)
                X1, X2 = self.select_(data, xTrain, kernel_type, subset_all[i], feature_indices[i,:], sample_indices[i,:])
                data1 = self.kernel_transform( X1 = X1, X2 = X2, kernel_type = kernel_type, n_components = n_components, gamma = gamma)
                data1 = self.normalize_(data1,me_all[i],std_all[i])
                M = data1.shape[1]
                data1 = self.add_bias(data1)                        
                
                W = W_all[i]  
                
                
                if(self.problem_type == 'regression'):
                    scores = np.zeros((data1.shape[0],numClasses))
                    for j in range(numClasses):
                        W2 = W[:,j:j+2]
                        W1 = (W2[:,0] - W2[:,1])/2
                        scores1 = data1[:,0:M].dot(W1[0:M,]) + np.dot(data1[:,M], W1[M+1,])
                        scores1 = -1.0/(W1[M,] + 1e-08)*scores1
                        scores[:,j] = scores1
                
#                if(self.problem_type == 'regression'):
#                    W1 = (W[:,0]-W[:,1])/2
#                    scores=data1[:,0:M].dot(W1[0:M,]) + np.dot(data1[:,M], W1[M+1,])
#                    scores = -1.0/(W1[M,] + 1e-08)*scores
#                    P_all[:,i] = scores
                    
                if(self.problem_type == 'classification'):
                    scores = data1.dot(W)
                    
                P_all[:,i*numClasses:numClasses+i*numClasses] = scores
                    
            if(n_ensembles == 1):
                if(self.problem_type == 'regression'):
                    if(numClasses == 1):
                        label = np.reshape(P_all,(P_all.shape[0],))
                    else:
                        label = P_all
                if(self.problem_type == 'classification'):
                    label=np.argmax(P_all,axis=1)
                return label
            
            W = W_all[n_ensembles]
            M = P_all.shape[1]            
#            P_all = self.add_bias(P_all)

            if(self.problem_type == 'regression'):
                scores = np.zeros((P_all.shape[0],numClasses))
                P_all_1 = np.zeros((P_all.shape[0],n_ensembles))
#                W = np.zeros((P_all_1.shape[1]+2,numClasses*2))
                for j in range(numClasses):
                    P_all_1 = np.zeros((P_all.shape[0],n_ensembles))
                    for k in range(n_ensembles):
                        P_all_1[:,k] = P_all[:,numClasses*k+j]
                    M = P_all_1.shape[1]  
                    P_all_1 = self.add_bias(P_all_1)
                    W2 = W[:,j:j+2]
                    W1 = (W2[:,0] - W2[:,1])/2
                    scores1 = P_all_1[:,0:M].dot(W1[0:M,]) + np.dot(P_all_1[:,M], W1[M+1,])
                    scores1 = -1.0/(W1[M,] + 1e-08)*scores1
                    scores[:,j] = scores1
                label = scores
                return label
#                    W1 = (W[:,0]-W[:,1])/2
#                    scores=P_all[:,0:M].dot(W1[0:M,]) + np.dot(P_all[:,M], W1[M+1,])
#                    scores = -1.0/(W1[M,] + 1e-08)*scores
#                    label = scores
                    
            if(self.problem_type == 'classification'):
                P_all = self.add_bias(P_all)
                scores = P_all.dot(W)
                label = np.argmax(scores,axis=1)        
                return label   
    
    def accuracy_classifier(self,actual_label,found_labels):
        acc=np.divide(np.sum(actual_label==found_labels)*100.0 , actual_label.shape[0],dtype='float64')
        return acc
    
    def accuracy_regressor(self,actual_label,found_labels):
        acc=np.divide(np.linalg.norm(actual_label - found_labels)**2 , actual_label.shape[0],dtype='float64')
        return acc
        
    
    def train_LSMCM(self, xTrain, yTrain, level, K_plus = None, K_minus = None, W = None):
        #min D(E|w|_1 + (1-E)*0.5*|W|_2^2) + C*\sum_i\sum_(j)|f_j(i)**2| + \sum_i\sum_(j_\neq y_i)(1-f_y_i(i) + f_j(i))**2
        #setting C = 0 gives us SVM
        # or when using margin term i.e., reg_type = 'M'
        #min D(E|w|_1) + (E)*0.5*\sum_j=1 to numClasses (w_j^T(K+ - K-)w_j) + C*\sum_i\sum_(j)|f_j(i)**2| + \sum_i\sum_(j_\neq y_i)(1-f_y_i(i) + f_j(i))**2
        #setting C = 0 gives us SVM with margin term
#        print('LSMCM Training')
#        print('reg_type=%s, algo_type=%s, problem_type=%s,kernel_type=%s'%(self.reg_type,self.algo_type,self.problem_type,self.kernel_type))
#        print('C1=%0.4f, C2=%0.4f, C3=%0.4f'%(self.C1,self.C2,self.C3))
        if(self.upsample1==True):
            xTrain,yTrain=self.upsample(xTrain,yTrain,new_imbalance_ratio=0.5,upsample_type=1)
            
        xTrain=self.add_bias(xTrain)
        
        M=xTrain.shape[1]
        N=xTrain.shape[0]
        numClasses=np.unique(yTrain).size
        verbose = False
        if(level==0):
            C = self.C1 #for loss function of MCM
            D = self.C2 #for L1 or L2 penalty
            E = self.C3 #for elastic net penalty or margin term
        else:
            C = self.C4 #for loss function of MCM 
            D = self.C2 #for L1 or L2 penalty
            E = self.C3 #for elastic net penalty since in combining the classifiers we use a linear primal classifier
            
        iterMax1 = self.iterMax1
        eta_zero = self.eta
        class_weighting = self.class_weighting
        reg_type = self.reg_type
        update_type = self.update_type
        tol = self.tol
        np.random.seed(1)
        
        if(W is None):
            W=0.001*np.random.randn(M,numClasses)
            W=W/np.max(np.abs(W))
        else:
            W_orig = np.zeros(W.shape)
            W_orig[:] = W[:]
        
        class_weights=np.zeros((numClasses,))
        sample_weights=np.zeros((N,))
        #divide the data into K clusters
    
        for i in range(numClasses):
            idx=(yTrain==i)           
            class_weights[i]=1.0/np.sum(idx)
            sample_weights[idx]=class_weights[i]
                        
        G_clip_threshold = 100
        W_clip_threshold = 500
        eta=eta_zero
                       
        scores = xTrain.dot(W) #samples X numClasses
        N = scores.shape[0]
        correct_scores = scores[range(N),np.array(yTrain,dtype='int32')]
        mat = (scores.transpose()-correct_scores.transpose()).transpose() 
        mat = mat+1.0
        mat[range(N),np.array(yTrain,dtype='int32')] = 0.0
        
        scores1  = np.zeros(scores.shape)
        scores1[:] = scores[:]
        scores1[range(N),np.array(yTrain,dtype='int32')] = -np.inf
        max_scores = np.max(scores1,axis =1)
        mat1 = 1 - correct_scores + max_scores
#        thresh1 = np.zeros(mat.shape)
#        thresh1[mat>0.0] = mat[mat>0.0] #for the SVM loss 
        #(1- f_yi + max_j neq yi f_j)^2
        f=0.0
        if(reg_type=='l2'):
            f += D*0.5*np.sum(W**2) 
        if(reg_type=='l1'):
            f += D*np.sum(np.abs(W))
        if(reg_type=='en'):
            f += D*0.5*(1-E)*np.sum(W**2)  +  D*E*np.sum(np.abs(W))
            
            
        if(class_weighting=='average'):
            f1 = C*0.5*np.sum(scores**2) + 0.5*np.sum((mat1)**2)
            f += (1.0/N)*f1 
        else:
            f1 = C*0.5*np.sum((scores**2)*sample_weights[:,None]) + 0.5*np.sum((mat1**2)*sample_weights[:,None])
            f+= (1.0/numClasses)*f1 
        
        if(K_minus is not None):
            temp_mat = np.dot(K_minus,W_orig[0:(M-1),])        
        
        for i in range(numClasses):
            #add the term (E/2*numclasses)*lambda^T*K_plus*lambda for margin
            if(K_plus is not None):
                w = W[0:(M-1),i]
                f2 = np.dot(np.dot(K_plus,w),w)
                f+= ((0.5*E)/(numClasses))*f2  
             #the second term in the objective function
            if(K_minus is not None):
                f3 = np.dot(temp_mat[:,i],w)
                f+= -((0.5*E)/(numClasses))*f3
        
        
        iter1=0
        print('iter1=%d, f=%0.3f'%(iter1,f))
                
        f_best=f
        fvals=np.zeros((iterMax1+1,))
        fvals[iter1]=f_best
        W_best=np.zeros(W.shape)
        iter_best=iter1
        f_prev=f_best
        rel_error=1.0
#        f_prev_10iter=f
        
        if(reg_type=='l1' or reg_type =='en' or reg_type == 'M'):
            # from paper: Stochastic Gradient Descent Training for L1-regularized Log-linear Models with Cumulative Penalty
            if(update_type == 'adam' or update_type == 'adagrad' or update_type == 'rmsprop'):
                u = np.zeros(W.shape)
            else:
                u = 0.0
            q=np.zeros(W.shape)
            z=np.zeros(W.shape)
            all_zeros=np.zeros(W.shape)
        
        eta1=eta_zero 
        v=np.zeros(W.shape)
        v_prev=np.zeros(W.shape)    
        vt=np.zeros(W.shape)
        m=np.zeros(W.shape)
        vt=np.zeros(W.shape)
        
        cache=np.zeros(W.shape)
        eps=1e-08
        decay_rate=0.99
        mu1=0.9
        mu=mu1
        beta1 = 0.9
        beta2 = 0.999  
        iter_eval=10 #evaluate after every 10 iterations
        
        idx_batches, sample_weights_batch, num_batches = self.divide_into_batches_stratified(yTrain)
        while(iter1<iterMax1 and rel_error>tol):
            iter1=iter1+1            
            for batch_num in range(0,num_batches):
    #                batch_size=batch_sizes[j]
                test_idx=idx_batches[batch_num]
                data=xTrain[test_idx,]
                labels=yTrain[test_idx,] 
                N=labels.shape[0]
                scores=data.dot(W)
                correct_scores=scores[range(N),np.array(labels,dtype='int32')]#label_batches[j] for this line should be in the range [0,numClasses-1]
                mat=(scores.transpose()-correct_scores.transpose()).transpose() 
                mat=mat+1.0
                mat[range(N),np.array(labels,dtype='int32')]=0.0                
                
                scores1  = np.zeros(scores.shape)
                scores1[:] = scores[:]
                scores1[range(N),np.array(labels,dtype='int32')] = -np.inf
                max_scores = np.max(scores1,axis =1)
                max_scores_idx = np.argmax(scores1, axis = 1)
                mat1 = 1 - correct_scores + max_scores                
                
                dscores1 = np.zeros(mat.shape)
                dscores1[range(N),np.array(max_scores_idx,dtype='int32')] = mat1
                row_sum = np.sum(dscores1,axis=1)
                dscores1[range(N),np.array(labels,dtype='int32')] = -row_sum
                
                if(C !=0.0):
                    dscores2 = np.zeros(scores.shape)
                    dscores2[:] = scores[:]
                else:
                    dscores2 = 0
                    
                dscores1 = 2*dscores1
                dscores2 = 2*dscores2
                if(class_weighting=='average'):
                    gradW = np.dot((dscores1 + C*dscores2).transpose(),data)
                    gradW = gradW.transpose()
                    gradW = (0.5/N)*gradW
#                    gradW += gradW1 - gradW2
                else:
                    sample_weights_b = sample_weights_batch[batch_num]
                    gradW = np.dot((dscores1 + C*dscores2).transpose(),data*sample_weights_b[:,None])
                    gradW = gradW.transpose()
                    gradW = (0.5/numClasses)*gradW
#                    gradW += gradW1 - gradW2
                        
                if(np.sum(gradW**2)>G_clip_threshold):#gradient clipping
#                    print('clipping gradients')
                    gradW = G_clip_threshold*gradW/np.sum(gradW**2)
                    
                if(update_type=='sgd'):
                    W = W - eta*gradW
                elif(update_type=='momentum'):
                    v = mu * v - eta * gradW # integrate velocity
                    W += v # integrate position
                elif(update_type=='nesterov'):
                    v_prev[:] = v[:] # back this up
                    v = mu * v - eta * gradW # velocity update stays the same
                    W += -mu * v_prev + (1 + mu) * v # position update changes form
                elif(update_type=='adagrad'):
                    cache += gradW**2
                    W += - eta1* gradW / (np.sqrt(cache) + eps)
                elif(update_type=='rmsprop'):
                    cache = decay_rate * cache + (1 - decay_rate) * gradW**2
                    W += - eta1 * gradW / (np.sqrt(cache) + eps)
                elif(update_type=='adam'):
                    m = beta1*m + (1-beta1)*gradW
                    mt = m / (1-beta1**(iter1+1))
                    v = beta2*v + (1-beta2)*(gradW**2)
                    vt = v / (1-beta2**(iter1+1))
                    W += - eta1 * mt / (np.sqrt(vt) + eps)           
                else:
                    W = W - eta*gradW
                    
                if(reg_type == 'M'):
                    gradW1= np.zeros(W.shape)
                    gradW2= np.zeros(W.shape)
                    for i in range(numClasses):
                        w=W[0:(M-1),i]
                        if(K_plus is not None):
                            gradW1[0:(M-1),i]=((E*0.5)/(numClasses))*2*np.dot(K_plus,w)
                        if(K_minus is not None):
                            gradW2[0:(M-1),i]=((E*0.5)/(numClasses))*temp_mat[:,i]
                    if(update_type == 'adam'):
                        W += -(gradW1-gradW2)*(eta1/(np.sqrt(vt) + eps)) 
                    elif(update_type == 'adagrad' or update_type =='rmsprop'):
                        W += -(gradW1-gradW2)*(eta1/(np.sqrt(cache) + eps))
                    else:
                        W += -(gradW1-gradW2)*(eta)
                        
                if(reg_type == 'ISTA'):
                    if(update_type == 'adam'):
                        idx_plus =  W > D*(eta1/(np.sqrt(vt) + eps))
                        idx_minus = W < -D*(eta1/(np.sqrt(vt) + eps))
                        idx_zero = np.abs(W) < D*(eta1/(np.sqrt(vt) + eps))
                        W[idx_plus] = W[idx_plus] - D*(eta1/(np.sqrt(vt[idx_plus]) + eps))
                        W[idx_minus] = W[idx_minus] + D*(eta1/(np.sqrt(vt[idx_minus]) + eps))
                        W[idx_zero] = 0.0
                    elif(update_type == 'adagrad' or update_type =='rmsprop'):
                        idx_plus =  W > D*(eta1/(np.sqrt(cache) + eps))
                        idx_minus = W < -D*(eta1/(np.sqrt(cache) + eps))
                        idx_zero = np.abs(W) < D*(eta1/(np.sqrt(cache) + eps))
                        W[idx_plus] = W[idx_plus] - D*(eta1/(np.sqrt(cache[idx_plus]) + eps))
                        W[idx_minus] = W[idx_minus] + D*(eta1/(np.sqrt(cache[idx_minus]) + eps))
                        W[idx_zero] = 0.0
                    else:
                        idx_plus =  W > D*(eta)
                        idx_minus = W < -D*(eta)
                        idx_zero = np.abs(W) < D*(eta)
                        W[idx_plus] = W[idx_plus] - D*(eta)
                        W[idx_minus] = W[idx_minus] + D*(eta)
                        W[idx_zero] = 0.0

                        
                if(reg_type=='l2'):
                    if(update_type == 'adam'):
                        W += -D*W*(eta1/(np.sqrt(vt) + eps)) 
                    elif(update_type == 'adagrad' or update_type =='rmsprop'):
                        W += -D*W*(eta1/(np.sqrt(cache) + eps))
                    else:
                        W += -D*W*(eta)  
                
                if(reg_type=='en'):
                    if(update_type == 'adam'):
                        W += -D*(1.0-E)*W*(eta1/(np.sqrt(vt) + eps)) 
                    elif(update_type == 'adagrad' or update_type =='rmsprop'):
                        W += -D*(1.0-E)*W*(eta1/(np.sqrt(cache) + eps))
                    else:
                        W += -D*W*(eta)  
                    
                if(reg_type=='l1' or reg_type == 'M'):
                    if(update_type=='adam'):
                        u = u + D*(eta1/(np.sqrt(vt) + eps))
                    elif(update_type == 'adagrad' or update_type =='rmsprop'):
                        u = u + D*(eta1/(np.sqrt(cache) + eps))
                    else:
                        u = u + D*eta
                    z[:] = W[:]
                    idx_plus = W>0
                    idx_minus = W<0
                    
                    W_temp = np.zeros(W.shape)
                    if(update_type=='adam' or update_type == 'adagrad' or update_type =='rmsprop'):
                        W_temp[idx_plus]=np.maximum(all_zeros[idx_plus],W[idx_plus]-(u[idx_plus]+q[idx_plus]))
                        W_temp[idx_minus]=np.minimum(all_zeros[idx_minus],W[idx_minus]+(u[idx_minus]-q[idx_minus]))                    
                    else:
                        W_temp[idx_plus]=np.maximum(all_zeros[idx_plus],W[idx_plus]-(u+q[idx_plus]))
                        W_temp[idx_minus]=np.minimum(all_zeros[idx_minus],W[idx_minus]+(u-q[idx_minus]))
                    
                    W[idx_plus]=W_temp[idx_plus]
                    W[idx_minus]=W_temp[idx_minus]
                    q=q+(W-z)
                    
                if(reg_type=='en'):
                    if(update_type=='adam'):
                        u = u + D*E*(eta1/(np.sqrt(vt) + eps))
                    elif(update_type == 'adagrad' or update_type =='rmsprop'):
                        u = u + D*E*(eta1/(np.sqrt(cache) + eps))                    
                    else:
                        u = u + D*E*eta
                    z[:] = W[:]
                    idx_plus = W>0
                    idx_minus = W<0
                    
                    W_temp = np.zeros(W.shape)
                    if(update_type=='adam' or update_type == 'adagrad' or update_type =='rmsprop'):
                        W_temp[idx_plus]=np.maximum(all_zeros[idx_plus],W[idx_plus]-(u[idx_plus]+q[idx_plus]))
                        W_temp[idx_minus]=np.minimum(all_zeros[idx_minus],W[idx_minus]+(u[idx_minus]-q[idx_minus]))                    
                    else:
                        W_temp[idx_plus]=np.maximum(all_zeros[idx_plus],W[idx_plus]-(u+q[idx_plus]))
                        W_temp[idx_minus]=np.minimum(all_zeros[idx_minus],W[idx_minus]+(u-q[idx_minus]))
                    W[idx_plus]=W_temp[idx_plus]
                    W[idx_minus]=W_temp[idx_minus]
                    q=q+(W-z)
                
                if(np.sum(W**2)>W_clip_threshold):#gradient clipping
#                    print('clipping normW')
                    W = W_clip_threshold*W/np.sum(W**2)
            
            if(iter1%iter_eval==0):                    
                #once the W are calculated for each epoch we calculate the scores
                scores=xTrain.dot(W)
#                scores=scores-np.max(scores)
                N=scores.shape[0]
                correct_scores = scores[range(N),np.array(yTrain,dtype='int32')]
                mat = (scores.transpose()-correct_scores.transpose()).transpose() 
                mat = mat+1.0
                mat[range(N),np.array(yTrain,dtype='int32')] = 0.0
#                thresh1 = np.zeros(mat.shape)
#                thresh1[mat>0.0] = mat[mat>0.0] #for the SVM loss 
                scores1  = np.zeros(scores.shape)
                scores1[:] = scores[:]
                scores1[range(N),np.array(yTrain,dtype='int32')] = -np.inf
                max_scores = np.max(scores1,axis =1)
                mat1 = 1 - correct_scores + max_scores
                
                f=0.0
                if(reg_type=='l2'):
                    f += D*0.5*np.sum(W**2) 
                if(reg_type=='l1'):
                    f += D*np.sum(np.abs(W))
                if(reg_type=='en'):
                    f += D*0.5*(1-E)*np.sum(W**2)  +  D*E*np.sum(np.abs(W))
                  
                if(class_weighting=='average'):
                    f1 = C*0.5*np.sum(scores**2) + 0.5*np.sum(mat1**2)
                    f += (1.0/N)*f1 
                else:
                    f1 = C*0.5*np.sum((scores**2)*sample_weights[:,None]) + 0.5*np.sum((mat1**2)*sample_weights[:,None])
                    f+= (1.0/numClasses)*f1 
                    
                for i in range(numClasses):
                    #first term in objective function for margin
                    if(K_plus is not None):
                        w = W[0:(M-1),i]
                        f2 = np.dot(np.dot(K_plus,w),w)
                        f += ((0.5*E)/(numClasses))*f2  
                        #the second term in the objective function for margin
                    if(K_minus is not None):
                        f3 = np.dot(temp_mat[:,i],w)
                        f += -((0.5*E)/(numClasses))*f3
                        
                if(verbose == True):        
                    print('iter1=%d, f=%0.3f'%(iter1,f))
                    
                fvals[iter1]=f
                rel_error=np.abs(f_prev-f)/np.abs(f_prev)
                max_W = np.max(np.abs(W))
                W[np.abs(W)<1e-03*max_W]=0.0
                
                if(f<f_best):
                    f_best=f
                    W_best[:]=W[:]
                    max_W = np.max(np.abs(W))
                    W_best[np.abs(W_best)<1e-03*max_W]=0.0
                    iter_best=iter1
                else:
                    break
                f_prev=f      
 
            eta=eta_zero/np.power((iter1+1),1)
            
        fvals[iter1]=-1
        return W_best,f_best,iter_best,fvals
    
