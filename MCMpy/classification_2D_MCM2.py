# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 15:41:59 2016

@author: mayank
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 18:41:34 2016

@author: mayank
original classes = S
subclasses = K
f_{abn}=w_{ij}x_n
E-step
min 0.5 \summation_{i=1^S}\summation_{j=1^K} (\|w_{ij}\|)^2 - 
C \summation_{n=1^N} \summation_{i=1^S} \summation_{j=1^K} y_{ijn} 
\bigg[ {f_{ijn}}) - \log(\summation_{p=1^S}\summation_{q=1^K}\exp^{f_{pqn}}) \bigg]
M-step
y_{ijn} = y_{in}P_{ijn}/P_{in}
"""
#plot decision boundary
import numpy as np
import pandas as pd
from time import time
from sklearn.model_selection import StratifiedKFold
import os
from sklearn.cluster import KMeans
from sklearn import datasets
import matplotlib.pyplot as plt


from scipy.stats import mode
#from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.utils import resample
from sklearn.neighbors import NearestNeighbors
from numpy.matlib import repmat
#from sklearn.covariance import OAS,LedoitWolf
#%%
hpc=False
print (os.getcwd())
if(hpc==False):
    path1="C:\\Users\\Skyler\\OneDrive\\IIT_Delhi\\Jayadeva\\MCM\\MCMpy"
else:
    path1="/home/ee/phd/eez142368/classification_datasets/label_partition/L1_SM_SGD_lin_label_partition_EM_Random_avg_large"
os.chdir(path1)
print (os.getcwd())
from MCM import MCM
#%%
def standardize(xTrain):
    me=np.mean(xTrain,axis=0)
    std_dev=np.std(xTrain,axis=0)
    #remove columns with zero std
    idx=(std_dev!=0.0)
    print(idx.shape)
    xTrain[:,idx]=(xTrain[:,idx]-me[idx])/std_dev[idx]
    return xTrain,me,std_dev

    #%%


#plt.figure()
#plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y)
datapath=path1 +'/data'
#randomly sample class=1
imbalance_ratio=1
#dataset_name=10
dataset_type='clustering'

for dataset_name in [1,2,4,5,6,8]:
#for imbalance_ratio in imbalance_ratio1:
#    dataset_name=dataset_name+1    
#    samples=5000
    typeAlgo= 'MCM_C'
    np.random.seed(1)
#    i=100
    #dataset_name='circles_%s'%(typeAlgo)    
#    samples = 1000
#    data = datasets.make_classification(n_samples=samples,n_features=2, n_redundant=0, n_informative=2,
#                                 n_clusters_per_class=1,random_state=1,flip_y=0.01, class_sep=2, hypercube=True,weights=None)    
##    data1 = datasets.make_classification(n_samples=samples,n_features=2, n_redundant=0, n_informative=2,
##                                 n_clusters_per_class=1,random_state=1,flip_y=0.0, class_sep=2, hypercube=True,weights=None)    
##    data=datasets.make_circles(n_samples=samples, shuffle=True, noise=0.1, random_state=i, factor=0.5)  
#    X = data[0]
#    Y = data[1]
    data=np.loadtxt(datapath+'/%d.txt'%(dataset_name))
    X=data[:,0:2]
    Y=data[:,2]
    Y=Y-1
    Y=np.array(Y,dtype=np.int32)
#    X,Y=upsample(X,Y,new_imbalance_ratio=5,upsample_type=2)
    #%%
#    c=0
#    plt.figure()
#    plt.scatter(X[Y==c, 0], X[Y==c, 1], marker='o', c=Y[Y==c])    
#    plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y)
    #%% standardize X
    X,me,std_dev=standardize(X)
#    plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y)
    #%%
    num_batches=5
    skf=StratifiedKFold(n_splits=num_batches, random_state=1, shuffle=True)
    j=0
    for train_index, test_index in skf.split(np.zeros(Y.shape[0],), Y):
        if(j==0):
            X1=X[train_index,]
            Y1=Y[train_index,]
            xTest=X[test_index,]
            yTest=Y[test_index,]
        j=j+1
        if(j==1):
            break
         
    num_batches=4
    skf=StratifiedKFold(n_splits=num_batches, random_state=1, shuffle=True)
    j=0
    for train_index, test_index in skf.split(np.zeros(Y1.shape[0],), Y1):
        if(j==0):
            xTrain=X1[train_index,]
            yTrain=Y1[train_index,]
            xVal=X1[test_index,]
            yVal=Y1[test_index,]
        j=j+1
        if(j==1):
            break
        
#    xVal=xTrain
#    yVal=yTrain
    #%%
    h = .02  # step size in the mesh    
    # create a mesh to plot in
    x_min, x_max = xTrain[:, 0].min() - 1, xTrain[:, 0].max() + 1
    y_min, y_max = xTrain[:, 1].min() - 1, xTrain[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h)) 
    #%%
    # parameter 'xyz' indices are denoted by 'xyz_idx' and it saves the indices, instead of strings in 'xyz' to be saved in a pandas dataframe
    # when running the functions the parameter 'xyz = 'abc' eg. kernel_type = 'rbf' can be passed as is 
    # the parameter indices eg. 'xyz_idx' : kernel_type_idx is not required unless you wish to save the results in a numpy array as I have
#    Ca = [0,1e-05,1e-03,1e-02,1e-01,1] #hyperparameter 1 #loss function parameter
    Ca = [0]
    Cb = [1e-04,1e-03,1e-02,1e-01,1,10] #hyperparameter 2 #when using L1 or L2 or ISTA penalty
#    Cb = [1e-04] #hyperparameter 2 #when using L1 or L2 or ISTA penalty

    Cc = [0,1e-04,1e-03,1e-02,1e-01,1,10] #hyperparameter 2 #when using elastic net penalty (this parameter should be between 0 and 1)
    Cc = [0] #hyperparameter 2 #when using elastic net penalty (this parameter should be between 0 and 1)
#   
#    Cc = [1e-04] #hyperparameter 2 #when using elastic net penalty (this parameter should be between 0 and 1)

    Cd = [0] #hyperparameter for final regressor or classifier used to ensemble when concatenating 
#        the outputs of previos layer of classifier or regressors
    problem_type1 = {0:'classification', 1:'regression'} #{0:'classification', 1:'regression'}
    problem_type = 'classification'
    problem_type_idx = 0
    algo_type1 = {0:'MCM',1:'LSMCM'}
    algo_type = 'MCM'
    algo_type_idx = 0
    kernel_type1 = {0:'linear', 1:'rbf', 2:'sin', 3:'tanh', 4:'TL1', 5:'linear_primal', 6:'rff_primal', 7:'nystrom_primal'} #{0:'linear', 1:'rbf', 2:'sin', 3:'tanh', 4:'TL1', 5:'linear_primal', 6:'rff_primal', 7:'nystrom_primal'}
    kernel_type = 'linear_primal'
    kernel_type_idx = 5
    gamma1 = [1e-04,1e-03,1e-02,1e-01,1,10,100] #hyperparameter3 (kernel parameter for non-linear classification or regression)
    gamma1 = [1] #hyperparameter3 (kernel parameter for non-linear classification or regression)

    epsilon1 = [0.0] #hyperparameter4 ( It specifies the epsilon-tube within which 
    #no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value.)
#    n_ensembles1 = [1]  #number of ensembles to be learnt, if setting n_ensembles > 1 then keep the sample ratio to be around 0.7
    n_ensembles = 1
#    feature_ratio1 = [1.0] #percentage of features to select for each PLM
    feature_ratio = 1.0
#    sample_ratio1 = [1.0] #percentage of data to be selected for each PLM
    sample_ratio = 1.0
#    batch_sz1 = [128] #batch_size
    batch_sz = 128
#    iterMax1a = [1000] #max number of iterations for inner SGD loop
    iterMax1 = 1000
    iterMax2 = 10
#    eta1 = [1e-02] #initial learning rate
    eta = 1e-02
#    tol1 = [1e-04] #tolerance to cut off SGD
    tol = 1e-04
    update_type1 =  {0:'sgd',1:'momentum',3:'nesterov',4:'rmsprop',5:'adagrad',6:'adam'}#{0:'sgd',1:'momentum',3:'nesterov',4:'rmsprop',5:'adagrad',6:'adam'}
    update_type ='adam'
    update_type_idx = 6
    reg_type1 = {0:'l1', 1:'l2', 2:'en', 4:'ISTA', 5:'M'} #{0:'l1', 1:'l2', 2:'en', 4:ISTA, 5:'M'}#ISTA: iterative soft thresholding (proximal gradient)
    reg_type = 'l1'
    reg_type_idx = 0
    feature_sel1 = {0:'sliding', 1:'random'} #{0:'sliding', 1:'random'}
    feature_sel = 'random'
    feature_sel_idx = 1
    class_weighting1 = {0:'average', 1:'balanced'}#{0:'average', 1:'balanced'}
    class_weighting = 'average'
    class_weighting_idx = 0
    combine_type1 =  {0:'concat',1:'average',2:'mode'} #{0:'concat',1:'average',2:'mode'}
    combine_type = 'average'
    combine_type_idx = 1
    upsample1a =  {0:False, 1:True} #{0:'False', 1:'True'}
    upsample1  = False
    upsample1_idx = 0
    PV_scheme1 = {0:'kmeans', 1:'renyi'}  #{0:'kmeans', 1:'renyi'}
    PV_scheme = 'renyi'
    PV_scheme_idx = 1
    n_components = int(5*np.sqrt(xTrain.shape[0]))
    do_pca_in_selection1 = {0:False,1:True} 
    do_pca_in_selection = False 
    do_pca_in_selection_idx = 0
    
    #iterMax1=1000
    result=np.zeros((1,33))
    for C1 in Ca:
        for C2 in Cb:
            for C3 in Cc:
                for C4 in Cd:
                    for gamma in gamma1:
                        for epsilon in epsilon1:
                            t0=time()
                            mcm = MCM(C1 = C1, C2 = C2, C3 = C3, C4 = C4, problem_type = problem_type, algo_type = algo_type, kernel_type = kernel_type, gamma = gamma, 
                                      epsilon = epsilon, feature_ratio = feature_ratio, sample_ratio = sample_ratio, feature_sel = feature_sel, 
                                      n_ensembles = n_ensembles, batch_sz = batch_sz, iterMax1 = iterMax1, iterMax2 = iterMax2, eta = eta, tol = tol, update_type = update_type, 
                                      reg_type = reg_type, combine_type = combine_type, class_weighting = class_weighting, upsample1 = upsample1,
                                      PV_scheme = PV_scheme, n_components = n_components, do_pca_in_selection = do_pca_in_selection )
                            W_all, sample_indices, feature_indices, me_all, std_all, subset_all = mcm.fit(xTrain,yTrain)
                            t1=time()
                            time_elapsed=t1-t0
                    
                            train_pred=mcm.predict(xTrain, xTrain, W_all, sample_indices, feature_indices, me_all, std_all, subset_all)
                            val_pred=mcm.predict(xVal, xTrain, W_all, sample_indices, feature_indices, me_all, std_all, subset_all)
                    
                            train_acc=mcm.accuracy_classifier(yTrain,train_pred)
                            val_acc=mcm.accuracy_classifier(yVal,val_pred)
                            train_f1= f1_score(yTrain, train_pred, average='weighted') 
                            val_f1 =f1_score(yVal, val_pred, average='weighted') 
                            print ('C1=%0.3f, C2=%0.3f -> train acc= %0.2f, val acc=%0.2f'%(C1,C2,train_acc,val_acc))
        #                                                                                        print('batch_sz=%d'%(batch_sz))
                            non_zero_weights=0
                            total_weights = 0
                            if(kernel_type == 'linear' or kernel_type =='rbf' or kernel_type =='sin' or kernel_type =='tanh' or kernel_type =='TL1'):
#                                print('here0')
                                for i in range(n_ensembles):
                                    W=W_all[i]
                                    W2=np.zeros(W.shape)
                                    W2[W!=0.0]=1
                                    W2 =np.sum(W2,axis=1)
                                    non_zero_weights+=np.sum(W2 != 0)                            
                                    total_weights += np.sum(W!=0)
                            else:
#                                print('here1')
                                for i in range(n_ensembles):
                                    W=W_all[i]
                                    non_zero_weights+=np.sum(W!=0)
                                    total_weights = non_zero_weights

                            result=np.append(result,np.array([[train_acc, val_acc, time_elapsed, non_zero_weights, C1, C2, C3, C4, problem_type_idx, kernel_type_idx, gamma,
                                                               epsilon, feature_ratio, sample_ratio, feature_sel_idx, n_ensembles, batch_sz, 
                                                               iterMax1, eta, tol, update_type_idx, reg_type_idx, combine_type_idx, class_weighting_idx, upsample1_idx ,train_f1,val_f1,
                                                               PV_scheme_idx, n_components, do_pca_in_selection_idx,iterMax2,total_weights,algo_type_idx]]),axis=0)

    
    result=result[1:,]
    #print result
    result_pd=pd.DataFrame(result,index=range(0,result.shape[0]),columns=['0_train_acc', '1_val_acc', '2_time_elapsed', 
                           '3_non_zero_weights', '4_C1', '5_C2', '6_C3', '7_C4', '8_problem_type_idx', '9_kernel_type_idx',
                           '10_gamma', '11_epsilon', '12_feature_ratio', '13_sample_ratio', '14_feature_sel_idx', 
                           '15_n_ensembles', '16_batch_sz', '17_iterMax1', '18_eta', '19_tol', '20_update_type_idx', 
                           '21_reg_type_idx', '22_combine_type_idx', '23_class_weighting_idx', '24_upsample1_idx',
                           '25_train_f1','26_val_f1','27_PV_scheme_idx', '28_n_components', '29_do_pca_in_selection_idx' ,
                           '30_iterMax2','31_total_weights','32_algo_type_idx'])
    result_pd.to_csv(path1+"/results/Train_results_%d_%s"%(dataset_name,algo_type)+"_%s.csv"%(dataset_type))
    max_acc=np.max(result[:,26])
    max_acc_idx=result[:,26]==max_acc
    min_sv=np.min(result[max_acc_idx,31])
    min_sv_idx=result[:,31]==min_sv
    best_val_idx=max_acc_idx*min_sv_idx
    best_val_acc=np.where(best_val_idx==True)[0][0]
    
    C1=result[best_val_acc,4]
    C2=result[best_val_acc,5]
    C3=result[best_val_acc,6]
    C4=result[best_val_acc,7]
    problem_type_idx = int(result[best_val_acc,8])#{0:'classification', 1:'regression'}
    problem_type = problem_type1[problem_type_idx]
    kernel_type_idx = int(result[best_val_acc,9]) #{0:'linear', 1:'rbf', 2:'sin', 3:'tanh', 4:'TL1'}
    kernel_type = kernel_type1[kernel_type_idx]
    gamma = result[best_val_acc,10] #hyperparameter3 (kernel parameter for non-linear classification or regression)
    epsilon = result[best_val_acc,11] #hyperparameter4 ( It specifies the epsilon-tube within which 
    #no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value.)
    #number of ensembles to be learnt, if setting n_ensembles > 1 then keep the sample ratio to be around 0.7
    feature_ratio = result[best_val_acc,12] #percentage of features to select for each PLM
    sample_ratio = result[best_val_acc,13] #percentage of data to be selected for each PLM
    feature_sel_idx = int(result[best_val_acc,14]) #{0:'sliding', 1:'random'}
    feature_sel = feature_sel1[feature_sel_idx]
    n_ensembles = int(result[best_val_acc,15]) 
    batch_sz= int(result[best_val_acc,16]) #batch_size
    iterMax1 = int(result[best_val_acc,17]) #max number of iterations for inner SGD loop
    eta = result[best_val_acc,18] #initial learning rate
    tol = result[best_val_acc,19]#tolerance to cut off SGD
    update_type_idx = int(result[best_val_acc,20])#{0:'sgd',1:'momentum',3:'nesterov',4:'rmsprop',5:'adagrad',6:'adam'}
    update_type = update_type1[update_type_idx]
    reg_type_idx = int(result[best_val_acc,21]) #{0:'l1', 1:'l2', 2:'en', 4:il2, 5:'ISTA'}#ISTA: iterative soft thresholding (proximal gradient)
    reg_type = reg_type1[reg_type_idx]    
    combine_type_idx = int(result[best_val_acc,22])#{0:'concat',1:'average',2:'mode'}
    combine_type = combine_type1[combine_type_idx]    
    class_weighting_idx = int(result[best_val_acc,23]) #{0:'average', 1:'balanced'}
    class_weighting = class_weighting1[class_weighting_idx]
    upsample1_idx  = int(result[best_val_acc,24]) #{0:'False', 1:'True'}
    upsample1 = upsample1a[upsample1_idx]
    PV_scheme_idx  = int(result[best_val_acc,27]) #{0:'kmeans', 1:'renyi'}
    PV_scheme = PV_scheme1[PV_scheme_idx]
    n_components = int(result[best_val_acc,28]) 
    do_pca_in_selection_idx  = int(result[best_val_acc,29]) #{0:'False', 1:'True'}
    do_pca_in_selection = do_pca_in_selection1[do_pca_in_selection_idx]
    iterMax2 = int(result[best_val_acc,30]) #max number of iterations for outer SGD loop
    algo_type_idx = int(result[best_val_acc,32])#{0:'MCM', 1:'LSMCM'}
    algo_type = algo_type1[algo_type_idx]
    #%%
    #testing the dataset
    print('Training and testing')    
    result1=np.zeros((1,33))
    t0=time()
    mcm = MCM(C1 = C1, C2 = C2, C3 = C3, C4 = C4, problem_type = problem_type, algo_type = algo_type, kernel_type = kernel_type, gamma = gamma, 
              epsilon = epsilon, feature_ratio = feature_ratio, sample_ratio = sample_ratio, feature_sel = feature_sel, 
              n_ensembles = n_ensembles, batch_sz = batch_sz, iterMax1 = iterMax1, iterMax2 = iterMax2, eta = eta, tol = tol, update_type = update_type, 
             reg_type = reg_type, combine_type = combine_type, class_weighting = class_weighting, upsample1 = upsample1,
             PV_scheme = PV_scheme, n_components = n_components, do_pca_in_selection = do_pca_in_selection )
    W_all, sample_indices, feature_indices, me_all, std_all, subset_all = mcm.fit(xTrain,yTrain)
    t1=time()
    time_elapsed=t1-t0

    train_pred=mcm.predict(xTrain, xTrain, W_all, sample_indices, feature_indices, me_all, std_all, subset_all)
    test_pred=mcm.predict(xTest, xTrain, W_all, sample_indices, feature_indices, me_all, std_all, subset_all)

    train_acc=mcm.accuracy_classifier(yTrain,train_pred)
    test_acc=mcm.accuracy_classifier(yTest,test_pred)
    train_f1= f1_score(yTrain, train_pred, average='weighted') 
    test_f1 =f1_score(yTest, test_pred, average='weighted') 
    
    print ('C1=%0.3f, C2=%0.3f -> train acc= %0.2f, test acc=%0.2f'%(C1,C2,train_acc,test_acc))
    
    non_zero_weights=0
    total_weights = 0
    if(kernel_type == 'linear' or kernel_type =='rbf' or kernel_type =='sin' or kernel_type =='tanh' or kernel_type =='TL1'):
        for i in range(n_ensembles):
            W=W_all[i]
            W2=np.zeros(W.shape)
            W2[W!=0.0]=1
            W2 =np.sum(W2,axis=1)
            non_zero_weights+=np.sum(W2!=0)
            total_weights += np.sum(W!=0)
    else:
        for i in range(n_ensembles):
            W=W_all[i]
            non_zero_weights+=np.sum(W!=0)
            total_weights = non_zero_weights
    
    result1=np.append(result1,np.array([[train_acc, test_acc, time_elapsed, non_zero_weights, C1, C2, C3, C4, problem_type_idx, kernel_type_idx, gamma,
                                         epsilon, feature_ratio, sample_ratio, feature_sel_idx, n_ensembles, batch_sz, 
                                         iterMax1, eta, tol, update_type_idx, reg_type_idx, combine_type_idx,                                          
                                         class_weighting_idx, upsample1_idx,train_f1,test_f1,PV_scheme_idx, 
                                         n_components, do_pca_in_selection_idx, iterMax2,total_weights,algo_type_idx]]),axis=0)
    
    
    result1=result1[1:,]
    #print result
    result_pd1=pd.DataFrame(result1,index=range(0,result1.shape[0]),columns=['0_train_acc', '1_test_acc', '2_time_elapsed', 
                           '3_non_zero_weights', '4_C1', '5_C2', '6_C3', '7_C4', '8_problem_type_idx', '9_kernel_type_idx',
                           '10_gamma', '11_epsilon', '12_feature_ratio', '13_sample_ratio', '14_feature_sel_idx', 
                           '15_n_ensembles', '16_batch_sz', '17_iterMax1', '18_eta', '19_tol', '20_update_type_idx', 
                           '21_reg_type_idx', '22_combine_type_idx', '23_class_weighting_idx', '24_upsample1_idx',
                           '25_train_f1','26_test_f1','27_PV_scheme_idx', '28_n_components', '29_do_pca_in_selection_idx',
                           '30_iterMax2','31_total_weights','32_algo_type_idx'])
    result_pd1.to_csv(path1+"/results/Test_results_%d_%s"%(dataset_name,algo_type)+"_%s.csv"%(dataset_type))
    #%% plotting the graph
    if(xTest.shape[1]==2):
        temp=xx.ravel().shape
        xTest1=np.c_[xx.ravel(), yy.ravel(),np.ones(temp)]
        Z=mcm.predict(xTest1,xTrain, W_all, sample_indices, feature_indices, me_all, std_all, subset_all)
        #Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        #plot the lines
        fig, ax1 = plt.subplots(figsize=(10, 7.5))
                    
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.rainbow, alpha=0.8)
#        for j in range(n_ensembles):
#            W = W_all[i]
#            for k in range(xTest1.shape[1]-1):
#                y = -(1.0/(W[1,k]+1e-08))*(xx[0,:]*W[0,k]+W[2,k])
#                plt.plot(xx[0,:],y)
#            W1 = (W[:,0] - W[:,1])/2.0
#            y = -(1.0/(W1[1]+1e-08))*(xx[0,:]*W1[0]+W1[2])
#            plt.plot(xx[0,:],y)
            
            
        # Plot also the training points
        
        plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.gnuplot)
        plt.xlabel('x1')
        plt.ylabel('y1')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        fig_title='%d_%s_%s'%(dataset_name,dataset_type,algo_type)
        plt.savefig(path1+"/graphs/"+fig_title+'.pdf', bbox_inches='tight')
        plt.close()
##    
    
        
