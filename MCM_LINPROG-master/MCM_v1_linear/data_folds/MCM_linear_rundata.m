clc;
clear all;

for dataset=29:29%change the number from 1 to 30 you can put this in a loop like this
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
%     degree(1)=1;
    % for i=2:5
    %     degree(i)=degree(i-1)+1;
    % end
    
    d_min=[];
    d_min(1)=1;
    for i=2:5
        d_min(i)=d_min(i-1)+0.5;
    end
    
    C1=[];
    C1(1)=1e-03;
    for i=2:4
        C1(i)=C1(i-1)*10;
    end
    
    C2=[];
    C2(1)=1e-03;
    for i=2:4
        C2(i)=C2(i-1)*10;
    end
    
    
    %also for polynomial kernel
    % beta=1;
    result=[];
    %similarly you can define beta here for RBF kernel and C for cost
    for a1=1:length(C1)
        for a2=1:length(C2)
            for a3=1:length(d_min)
                t1=[];
                t2=[];
                t3=[];
                
                
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
                    [trainAcc,testAcc,nsv,exit] = linear_MCM(xTrain,yTrain,xTest,yTest,C1(a1),C2(a2),d_min(a3));
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
                r=[avg1 avg2 avg3 std1 std2 std3 C1(a1) C2(a2) d_min(a3)];
                result=[result;r];
            end
        end
    end
    
     str1=sprintf('MCM_result_linear/result_dataset%d.txt',dataset);
     save(str1,'result','-ascii');
end
