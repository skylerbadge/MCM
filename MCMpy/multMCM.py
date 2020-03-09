import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import linear_kernel,rbf_kernel,manhattan_distances,polynomial_kernel,sigmoid_kernel
from sklearn.model_selection import train_test_split
from scipy.optimize import linprog


def kernelfunction(kernel_type,X1,X2,gamma = 1.0):
    """
    X1: n_samples1 X M
    X2: n_samples2 X M
    X: n_samples1 X n_samples2 : if kernel_type is non primal
    X: n_samples1 X n_components : if kernel_type is primal
    """
    if(kernel_type == 'linear'):
        X = linear_kernel(X1,X2)
    elif(kernel_type == 'rbf'):
        X = rbf_kernel(X1,X2,gamma)   
    elif(kernel_type == 'sigmoid'):
        X = sigmoid_kernel(X1,X2,-gamma) 
    elif(kernel_type == 'sin'):
        X = np.sin(gamma*manhattan_distances(X1,X2))
    elif(kernel_type == 'poly'):
        X = polynomial_kernel(X1,X2,gamma)
    else:        
        print('No kernel_type passed: using linear primal solver')
        X = X1
    return X


def mcm_linear_efs_multker(xTrain, yTrain, kerType, kerPara, Cparam, alpha):
    n,d = xTrain.shape
    x0 = np.concatenate((np.random.rand(n),np.random.rand(1),np.random.rand(1),np.random.rand(n)),axis = 0)
  
    K = np.zeros((n,n))
    for k in range(len(alpha)):
        K=K+alpha[k]*kernelfunction(kerType[k],xTrain,xTrain,gamma = kerPara[k])
  
    f = np.concatenate((np.zeros(n),np.zeros(1),np.ones(1),Cparam*np.ones(n)),axis = 0)

    a_eq = []
    b_eq = []
  
    multy = np.matmul(np.diag(yTrain.ravel()),K)
  
    a_ineq1= np.append(-multy,-yTrain.reshape(n,1),axis = 1)
    a_ineq1= np.append(a_ineq1,np.zeros((n,1)),axis = 1)
    a_ineq1= np.append(a_ineq1,-np.identity(n),axis = 1)
  
    a_ineq2= np.append(multy,yTrain.reshape(n,1),axis = 1)
    a_ineq2= np.append(a_ineq2,-np.ones((n,1)),axis = 1)
    a_ineq2= np.append(a_ineq2,np.identity(n),axis = 1)
  
    a_ineq = np.append(a_ineq1,a_ineq2,axis = 0)
  
    b_ineq = np.append(-np.ones((n,1)),np.zeros((n,1)),axis = 0)
  
    lb = np.full((n,1),-np.inf)
    lb =  np.append(lb,np.full((1,1),-np.inf),axis = 0)
    lb =  np.append(lb,np.ones((1,1)),axis = 0)
    lb =  np.append(lb,np.zeros((n,1)),axis = 0)
  
    ub = np.full((n,1),np.inf)
    ub =  np.append(ub,np.full((1,1), np.inf),axis = 0)
    ub =  np.append(ub,np.full((1,1), np.inf),axis = 0)
    ub =  np.append(ub,np.full((n,1), np.inf),axis = 0)

#   print(str(lb.shape)+"  "+ str(ub.shape))
    bounds = np.append(lb,ub,axis = 1)
  
    res = linprog(f, A_ub=a_ineq, b_ub=b_ineq, A_eq=None, b_eq=None, bounds=bounds, method='simplex', callback=None, options=None)
  
    if(res.success):
        return res.x[0:n],res.x[n],res.x[n+1]
    else:
        print("no 1 : ",res)
        return np.zeros(n),0,0


def mcm_ker_mult_v2(xTrain, yTrain, kerType, kerPara, Cparam, lmbda):
    n,d = xTrain.shape
    numKernels = len(kerType)
  
    x0 = np.concatenate((np.random.rand(numKernels),np.random.rand(n),np.random.rand(1),np.random.rand(1)),axis = 0)
  
    f = np.concatenate((np.zeros(numKernels),Cparam*np.ones(n),np.ones(1),np.zeros(1)),axis = 0)

#   print(lmbda.shape,"  ",yTrain.shape)
    a_ineq1 = -np.matmul(kernelfunction(kerType[0],xTrain,xTrain,gamma = kerPara[0]),yTrain*lmbda.reshape(n,1))
#   print(a_ineq1.shape)
    b_ineq1 = -np.ones((n,1))
  
    for i in range(1,numKernels):
        a_ineq1 = np.append(a_ineq1,-np.matmul(kernelfunction(kerType[i],xTrain,xTrain,gamma = kerPara[i]),yTrain*lmbda.reshape(n,1)),axis = 1)
  
    a_ineq1 = np.append(a_ineq1,-np.identity(n),axis = 1)
    a_ineq1 = np.append(a_ineq1,np.zeros((n,1)),axis = 1)
    a_ineq1 = np.append(a_ineq1,-yTrain.reshape(n,1),axis = 1)
  
    a_ineq2 = np.matmul(kernelfunction(kerType[0],xTrain,xTrain,gamma = kerPara[0]),yTrain*lmbda.reshape(n,1))
    b_ineq2 = np.ones((n,1))
  
    for i in range(1,numKernels):
        a_ineq2 = np.append(a_ineq2,np.matmul(kernelfunction(kerType[i],xTrain,xTrain,gamma = kerPara[i]),yTrain*lmbda.reshape(n,1)),axis = 1)
  
    a_ineq2 = np.append(a_ineq2,np.identity(n),axis = 1)
    a_ineq2 = np.append(a_ineq2,-np.ones((n,1)),axis = 1)
    a_ineq2 = np.append(a_ineq2,yTrain.reshape(n,1),axis = 1)
  
    a_ineq = np.append(a_ineq1,a_ineq2,axis = 0)
    b_ineq = np.append(b_ineq1,b_ineq2,axis = 0)
  
    a_eq = np.concatenate((np.ones(numKernels),np.zeros(n),np.zeros(1),np.zeros(1)),axis = 0).reshape(1,numKernels+n+2)
    b_eq = np.array([1]).reshape(1,1)
  
    lb = np.concatenate((np.zeros(numKernels),np.zeros(n),np.ones(1),np.full(1,-np.inf)),axis = 0).reshape(n+numKernels+2,1)
  
    ub = np.concatenate((np.ones(numKernels),np.full(n,np.inf),np.full(1,np.inf),np.full(1,np.inf)),axis = 0).reshape(n+numKernels+2,1)
 
    bounds = np.append(lb,ub,axis = 1)
  
    #print(a_ineq.shape,b_ineq.shape,a_eq.shape,b_eq.shape,bounds.shape)
  
    res = linprog(f, A_ub=a_ineq, b_ub=b_ineq, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method='simplex', callback=None, options=None)
  
    if(res.success):
        return res.x[0:numKernels]
    else:
        print("no v2 :", res)
    return np.zeros(numKernels)

def testMultiKernel( xTrain,xTest,yTest,kerType,kerPara,lmbda,b,alpha ):
    n,d = xTrain.shape
    ntest,dtest = xTest.shape
    pred = np.zeros(ntest)
  
    for i in range(ntest):
        K = np.zeros((n,1))
        for j in range(n):
            for k in range(len(alpha)):
                K[j]=K[j]+alpha[k]*kernelfunction(kerType[k],xTest[i,:].reshape(1,-1),xTrain[j,:].reshape(1,-1),gamma = kerPara[k])
    
    predVal = (np.matmul(np.transpose(lmbda),K)+b)/np.linalg.norm(lmbda)
#     print(predVal.shape)
    if(predVal>=0):
        pred[i] = 1
    else:
        pred[i] = -1
     
    acc = (np.sum(pred==yTest)/ntest)*100
    
    if(acc<50):
        pred = -pred
        acc = (np.sum(pred==yTest)/ntest)*100
    
    return acc


def tuneMCMmultker(xTrain, yTrain, kerTypeMCM , cParams , gamma , alpha ):
    n,d = xTrain.shape
    xTrain1,xValid,yTrain1,yValid = train_test_split(xTrain, yTrain, test_size = 0.2, random_state = 42)
    bestAcc = 0
    Cbest = 0
    tempalpha = alpha

    for i in range(len(cParams)):

        Ctest = cParams[i]

        iterMax = 10
        new_acc = 1
        old_acc = 0
        itr = 1

        alpha = tempalpha

        while itr<=iterMax and new_acc>old_acc :
            lmbda,b,h = mcm_linear_efs_multker(xTrain1, yTrain1, kerTypeMCM , gamma, Ctest, alpha)

        #       prevalpha = alpha

            alpha = mcm_ker_mult_v2(xTrain1, yTrain1, kerTypeMCM, gamma , Ctest, lmbda)

            acc = testMultiKernel(xTrain1,xValid,yValid,kerTypeMCM, gamma ,lmbda,b,alpha)

            old_acc = new_acc
            new_acc = acc

            itr+=1

            if new_acc>=bestAcc:
                bestAcc = new_acc
                Cbest = Ctest

    print("Tuning done C: ",Cbest," Acc ",bestAcc)

    return Cbest


def main():
    for dataset in [30]:
        print(dataset)
 
        data = pd.read_csv('data_folds/'+str(dataset)+'.csv')
        data = np.array(data)

        folds = pd.read_csv('data_folds/'+str(dataset)+'fold.csv')
        folds = np.array(folds).ravel()

        y = data[:,-1]
        x = data[:,:-1]

        nfolds = 5
        m = x.shape[0]

        kerTypeMCM = ['rbf','rbf']
        alpha = [0.2,0.2]
        gamma = [0.001,5]
        Cparams = [0.001,5]
    
#     kerTypeMCM = ['rbf','rbf','rbf','rbf','rbf']
#     alpha = [0.2,0.2,0.2,0.2,0.2]
#     gamma = [0.001,0.01,0.1,1,5]
#     Cparams = [0.001,0.01,0.1,1,5]
    
        results = pd.DataFrame(columns=['dataset', 'trainAcc', 'testAcc', 'nsv', 'stdTrainAcc', 'stdTestAcc', 'stdnsv', 'C', 'alpha1', 'alpha2', 'alpha3', 'alpha4', 'alpha5'])

        t1 = np.array([])
        t2 = np.array([])
        t3 = np.array([])


        for i in range(1,nfolds+1):
            bestAcc = 0
            finaltrainAcc = 0

            xTrain = x[np.argwhere(folds!=i).flatten()]
            yTrain = y[np.argwhere(folds!=i).flatten()]
            xTest = x[np.argwhere(folds==i).flatten()]
            yTest = y[np.argwhere(folds==i).flatten()]


            xTrain = (xTrain - (np.mean(xTrain,axis = 0)))/np.std(xTrain,axis = 0) if len(np.std(xTrain,axis = 0)==0)==0 else (xTrain - (np.mean(xTrain,axis = 0)))
            yTrain = (yTrain - (np.mean(yTrain,axis = 0)))/np.std(yTrain,axis = 0) if np.std(yTrain,axis = 0)!=0 else (xTrain - (np.mean(yTrain,axis = 0)))
            xTest = (xTest - (np.mean(xTest,axis = 0)))/np.std(xTest,axis = 0) if len(np.std(xTest,axis = 0)==0)==0 else (xTrain - (np.mean(xTest,axis = 0)))
            yTest = (yTest - (np.mean(yTest,axis = 0)))/np.std(yTest,axis = 0) if np.std(yTest,axis = 0)!=0 else (xTrain - (np.mean(yTest,axis = 0)))

            yTrain = yTrain.reshape(len(yTrain),1)
            yTest = yTest.reshape(len(yTest),1)

    #       print(yTrain)

            # alpha = [0.2,0.2,0.2,0.2,0.2]
            alpha = [0.2,0.2]

            Cbest = tuneMCMmultker(xTrain, yTrain, kerTypeMCM , Cparams , gamma , alpha)

            iterMax=10
            new_acc=1
            old_acc=0
            itr = 1

            nsv = 0

            while itr<=iterMax and new_acc>old_acc:

                lmbda,b,h = mcm_linear_efs_multker(xTrain, yTrain, kerTypeMCM , gamma, Cbest, alpha)

                alpha = mcm_ker_mult_v2( xTrain, yTrain, kerTypeMCM, gamma , Cbest, lmbda )

                acc = testMultiKernel(xTrain,xTrain,yTrain,kerTypeMCM, gamma ,lmbda,b,alpha)
                
                old_acc = new_acc
                new_acc = acc

                itr+=1

            testAcc = testMultiKernel(xTrain,xTest,yTest,kerTypeMCM,gamma,lmbda,b,alpha)
            trainAcc = testMultiKernel( xTrain,xTrain,yTrain,kerTypeMCM,gamma,lmbda,b,alpha );

            if testAcc >= bestAcc:
                bestAcc = testAcc
                finaltrainAcc = trainAcc
                bestalpha = alpha
                nsv = np.sum(lmbda>0.0001)

            np.append(t1,finaltrainAcc)
            np.append(t2,bestAcc)
            np.append(t3,nsv)

            print(np.mean(t2))

#         results = results.append({'dataset':dataset, 'trainAcc':np.mean(t1), 'testAcc':np.mean(t2), 'nsv':np.mean(t3), 'stdTrainAcc':np.std(t1), 'stdTestAcc':np.std(t2), 'stdnsv':np.std(t3), 'C':Cbest, 'alpha1':alpha[0], 'alpha2':alpha[1], 'alpha3':alpha[2], 'alpha4':alpha[3], 'alpha5':alpha[4]},ignore_index=True)
        print("done C,acc,alpha ",Cbest,np.mean(t2),alpha )
#     results.to_csv('results.csv')


if __name__ == "__main__":
    main()