function [trainAcc,testAcc,nsv,exit] = linear_MCM(xTrain,yTrain,xTest,yTest,C1,p)
    
    kerTypeMCM='rbf';
    kerParaMCM=p;
    Cparam=C1;
    
    [ lambda,b,h ] = mcm_linear_efs( xTrain, yTrain, kerTypeMCM, kerParaMCM, Cparam );
    
    [ test_pred,test_acc_mcm ] = mcmPredict( xTrain,xTest,yTest,kerTypeMCM,kerParaMCM,lambda,b );
    [ train_pred,train_acc_mcm ] = mcmPredict( xTrain,xTrain,yTrain,kerTypeMCM,kerParaMCM,lambda,b );
    
    trainAcc = train_acc_mcm;
    testAcc = test_acc_mcm;
    
    nsv = 0;
    exit = 1;
end