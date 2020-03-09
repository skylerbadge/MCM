clc;
clear all;

result=[];

for dataset=30:30%change the number from 1 to 30 you can put this in a loop like this
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
    
    kerTypeMCM={'rbf','rbf','rbf','rbf','rbf'};
    alpha=[0.2;0.2;0.2;0.2;0.2];
    gamma=[0.001;0.01;0.2;1;5];
    Cparams=[0.0001;0.001;0.02;0.1;5];
    
    
%     kerTypeMCM={'rbf'};
%     alpha=[0.2];
%     gamma=[0.001];
%     Cparams=[0.0001];

    %also for polynomial kernel
    % beta=1;
    %similarly you can define beta here for RBF kernel and C for cost
    
    t1=[];
    t2=[];
    t3=[];
    
    
    bestAcc=0;
    tic;
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
        
        
        [ Cbest ] = tuneMCMmultker( xTrain, yTrain, kerTypeMCM , Cparams , gamma , alpha );
        
        iterMax=10;
        new_acc=1;
        old_acc=0;
        iter=1;

        alpha=[0.2;0.2;0.2;0.2;0.2];
%         alpha=[0.2];
        nsv = 0;
        while (iter<=iterMax && new_acc>old_acc)
            try
                [ lambda,b,h ] = mcm_linear_efs_multker( xTrain, yTrain, kerTypeMCM , gamma, Cbest, alpha );
            catch
                lambda = rand(size(xTrain,1),1);
                b = rand;
            end

            try
                [ alpha ] = mcm_ker_mult_v2( xTrain, yTrain, kerTypeMCM, gamma , Cbest, lambda);    
            catch
                alpha = [0.2*rand;0.2*rand;0.2*rand;0.2*rand];
                alpha = [alpha ; 1-sum(alpha)];
                %assigning random and setting the last one such that the
                %sum is 1
            end

%             fprintf(2,'\n***********************************************************\n');
            [ pred,acc ] = testMultiKernel( xTrain,xTrain,yTrain,kerTypeMCM, gamma ,lambda,b,alpha );

%             fprintf(2,'Accuracy changed from %.3f to %.3f in iteration %d \n',old_acc,acc,iter);
%             alpha'

            % Swap accuracy variable
            temp=new_acc;
            new_acc=acc;
            old_acc=temp;

            % Increment iteration counter
            iter=iter+1;
        end

        [ test_pred,testAcc ] = testMultiKernel( xTrain,xTest,yTest,kerTypeMCM,gamma,lambda,b,alpha);
        [ train_pred,trainAcc ] = testMultiKernel( xTrain,xTrain,yTrain,kerTypeMCM,gamma,lambda,b,alpha );

        if(testAcc >= bestAcc)
                bestAcc = testAcc;
                bestalpha=alpha;
                nsv = length(nonzeros(lambda(lambda>1e-3)));
        end

% setting exit as 1 for testing    
        exit=1;
        
        if(exit==1)
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
    r=[dataset avg1 avg2 avg3 std1 std2 std3 Cbest alpha(1) alpha(2) alpha(3) alpha(4) alpha(5) timeFold];
    result=[result;r];
    
    best_acc = avg2;
    
    fprintf(2,'Best Accuracy :  %.3f     C: %.3f    A: %.3f,%.3f,%.3f,%.3f,%.3f',best_acc,Cbest,alpha(1),alpha(2),alpha(3),alpha(4),alpha(5));
end

dlmwrite('resultAlpha0_1.txt',result,'delimiter','\t','newline','pc','precision',5)
    
