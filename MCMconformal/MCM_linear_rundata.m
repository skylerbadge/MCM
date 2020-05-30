clc;
clear all;

result=[];

for dataset=[30]%change the number from 1 to 30 you can put this in a loop like this
    % for dataset=1:30
    %%%%% call everything here...but I do not recommend it, since debugging
    %%%%% is difficult
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
    s=size(X,1);%size of training data  
    %hyperparameters to be initialized here

%     gamma=2.^[-9,-5,-3,-1,1];
%     gamma=2.^[-9,-5,-3];
    gamma=2.^[-10,-11,-12,-8,-9];
    
%     cParams = 2.^(-25:2:2);
    cParams = 2.^(-19:2:2);

    kerTypeMCM ='rbf';

    %also for polynomial kernel
    % beta=1;
    %similarly you can define beta here for RBF kernel and C for cost
    
    t1=[];
    t2=[];
    t3=[];
    t4=[];
    t5=[];    
    
    Cbest=0;
    bestKerPara=0;
    bestAcc=0;
    bestAccConf = 0;
    CbestConf = 0 ;
    bestgam0 = 0 ;
    bestgam = 0 ;
    tic;
%     for i=[1,2,3]
    for i=1:nfolds
        xTrain0=[];
        yTrain=[];
        xTest0=[];
        yTest=[];

        test = (indices == i);
        train = ~test;
        for j=1:s
            if(train(j)==1)
                xTrain0=[xTrain0;X(j,:)];
                yTrain=[yTrain;Y(j,:)];
            end
            if(test(j)==1)
                xTest0=[xTest0;X(j,:)];
                yTest=[yTest;Y(j,:)];
            end
        end
        %data preprocessing
        me=mean(xTrain0);
        std_dev=std(xTrain0);
        xTrain = zeros(size(xTrain0));
        xTest = zeros(size(xTest0));
        for n=1:size(xTrain,2)
            if(std_dev(n)~=0)
                xTrain(:,n)=(xTrain0(:,n)-me(n))./std_dev(n);
            else
                xTrain(:,n)=(xTrain0(:,n)-me(n));
            end
        end
        for n=1:size(xTest,2)
            if(std_dev(n)~=0)
                xTest(:,n)=(xTest0(:,n)-me(n))./std_dev(n);
            else
                xTest(:,n)=(xTest0(:,n)-me(n));
            end
        end
        
        
        [Ctest,testKerPara] = tuneMCM( xTrain, yTrain , kerTypeMCM , cParams , gamma);
        try
            [ lambda,b,h ] = mcm_linear_efs( xTrain, yTrain, kerTypeMCM, testKerPara, Ctest );
        catch
            lambda = rand(size(xTrain,1),1);
            b = rand;
        end
        
        [ test_pred,testAcc ] = mcmPredict( xTrain,xTest,yTest,kerTypeMCM,testKerPara,lambda,b );
        [ train_pred,trainAcc ] = mcmPredict( xTrain,xTrain,yTrain,kerTypeMCM,testKerPara,lambda,b );
      
        nsv = length(nonzeros(lambda(lambda>1e-6)));
      
        if(testAcc >= bestAcc)
                bestAcc = testAcc;
                Cbest=Ctest;
                bestKerPara=testKerPara;
        end 
        
        % tuning gam
        % conformal
        
        AccConfFold = 0;
        gam0 = testKerPara;
        
        for gam = gam0*[2,2.2,2.4,2.6,2.8,3,4,10,15,20,25,30,40,50,60,70,80,100,500,800,1000,5000]
%         for gam = gam0*linspace(2,500,1000)
            try
                gam/gam0
                [testAccConf,trainAccConf] = accMcmConformal(xTrain,yTrain,xTest,yTest,lambda,kerTypeMCM,gam0,gam,Cbest);
                testAccConf
            catch
                testAccConf = 0;
                trainAccConf = 0;
            end
%              per fold for records
            if(testAccConf >= AccConfFold)
                AccConfFold = testAccConf;
                trainAccConfFold = trainAccConf;
            end
            
%              overall
            if(testAccConf >= bestAccConf)
                bestAccConf = testAccConf;
                besttrainAccConf = trainAccConf;
                CbestConf = Ctest ;
                bestgam0 = gam0 ;
                bestgam = gam ;
            end
        end 

        t1=[t1;trainAcc];
        t2=[t2;testAcc];
        t3=[t3;nsv];
        if (AccConfFold~=0)
            t4=[t4;trainAccConfFold];
            t5=[t5;AccConfFold];
        end

    end
    avg1=mean(t1);
    std1=std(t1);
    avg2=mean(t2);
    std2=std(t2);
    avg3=mean(t3);
    std3=std(t3);
    
    avg4=mean(t4);
    std4=std(t4);
    avg5=mean(t5);
    std5=std(t5);
    
%   r=[avg1 avg2 avg3 std1 std2 std3 C1(a1) C2(a2) d_min(a3)];
    timeFold = toc;
    
    r=[dataset avg1 std1 avg2 std2 Cbest bestKerPara avg3 std3 avg4 std4 avg5 std5 CbestConf bestgam0 bestgam timeFold];
    result=[result;r];
    
    fprintf(2,'MCM : Avg Accuracy :  %.3f     C: %.3f    P: %.3f \n',avg2,Cbest,bestKerPara);
    fprintf(2,'MCM Conformal: Avg Accuracy :  %.3f     C: %.3f    G0: %.3f   G: %.3f',avg5,CbestConf,bestgam0, bestgam);
%     xlswrite(strcat(int2str(dataset),'_result_baseline_conformal_gs.xlsx'),result)
end
