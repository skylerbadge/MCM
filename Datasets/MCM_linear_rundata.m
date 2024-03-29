clc;
clear all;

result=[];

for dataset=8%change the number from 1 to 30 you can put this in a loop like this
    % for dataset=1:30
    %%%%%%%% call everything here...but I do not recommend it, since debugging
    %%%%%%%% is difficult
    % end
    filename= sprintf('%d.mat',dataset);
    folds=sprintf('%dfold.mat',dataset);
    load(strcat('data_folds/',filename));
    load(strcat('data_folds/',folds));
    clc;
    disp(dataset);
    
    X=x;
    Y=y;
    nfolds=5;
    m=size(X,1);%size of training data
    %hyperparameters to be initialized here
    %degree for polynomial kernel
    % degree=[];
    % degree(1)=1;
    % for i=2:5
    %     degree(i)=degree(i-1)+1;
    % end
    
%     gamma=2.^[-9,-5,-3,-1,1];
    gamma=2.^[-3];
%     gamma=2.^[-9];
%     gamma(1)=0.1;
%     for i=2:5
%         gamma(i)=gamma(i-1)+0.25;
%     end
    
    cParams = 2.^(-25:2:2);
%     cParams = 2.^[-9];
    %     C1(1)=1e-03;
    %     for i=2:4
    %         C1(i)=C1(i-1)*10;
    %     end
    
    kerTypeMCM ='rbf';

    %also for polynomial kernel
    % beta=1;
    %similarly you can define beta here for RBF kernel and C for cost
    
    t1=[];
    t2=[];
    t3=[];
    
    Cbest=0;
    bestKerPara=0;
    bestAcc=0;
    tic;
%     for i=1
    for i=1:nfolds
        xTrain=[];
        yTrain=[];
        xTest=[];
        yTest=[];

        test = (indices == i);
        train = ~test;
        for j=1:m
            if(train(j)==1)
                xTrain=[xTrain;X(j,:)];
                yTrain=[yTrain;Y(j,:)];
            end
            if(test(j)==1)
                xTest=[xTest;X(j,:)];
                yTest=[yTest;Y(j,:)];
            end
        end
        %data preprocessing
        me=mean(xTrain);
        std_dev=std(xTrain);

        for n=1:size(xTrain,2)
            if(std_dev(n)~=0)
                xTrain(:,n)=(xTrain(:,n)-me(n))./std_dev(n);
            else
                xTrain(:,n)=(xTrain(:,n)-me(n));
            end
        end
        for n=1:size(xTest,2)
            if(std_dev(n)~=0)
                xTest(:,n)=(xTest(:,n)-me(n))./std_dev(n);
            else
                xTest(:,n)=(xTest(:,n)-me(n));
            end
        end
        %add your own MCM code here instead of lsMCMkernel_linprog
%       [trainAcc,testAcc,nsv,exit] = linear_MCM(xTrain,yTrain,xTest,yTest,C1(a1),C2(a2),d_min(a3));
        
        [Ctest,testKerPara] = tuneMCM( xTrain, yTrain , kerTypeMCM , cParams , gamma);
        try
            [ lambda,b,h ] = mcm_linear_efs( xTrain, yTrain, kerTypeMCM, testKerPara, Ctest );
        catch
            lambda = rand(size(xTrain,1),1);
            b = rand;
        end
        
        [ test_pred,testAcc ] = mcmPredict( xTrain,xTest,yTest,kerTypeMCM,testKerPara,lambda,b );
        [ train_pred,trainAcc ] = mcmPredict( xTrain,xTrain,yTrain,kerTypeMCM,testKerPara,lambda,b );
      
        nsv = length(nonzeros(lambda(lambda>1e-3)));
        
        if(testAcc >= bestAcc)
                bestAcc = testAcc;
                Cbest=Ctest;
                bestKerPara=testKerPara;
        end 
% setting exit as 1 for testing    
        exitFlag=1;
        
        if(exitFlag==1)
            t1=[t1;trainAcc];
            t2=[t2;testAcc];
            t3=[t3;nsv];
        end

    end
    avg1=mean(t1);
    avg2=mean(t2);
    avg3=mean(t3);
    std1=std(t1);
    std2=std(t2);
    std3=std(t3);
    
%   r=[avg1 avg2 avg3 std1 std2 std3 C1(a1) C2(a2) d_min(a3)];
    timeFold = toc;
    
    r=[dataset avg1 std1 avg2 std2 avg3 std3 Cbest bestKerPara timeFold];
    result=[result;r];
    bestAcc
    best_acc = avg2;
    
    fprintf(2,'Best Accuracy :  %.3f     C: %.3f    P: %.3f',best_acc,Cbest,bestKerPara);
    xlswrite(strcat(int2str(dataset),'result_baseline_gs.xlsx'),result)
end
