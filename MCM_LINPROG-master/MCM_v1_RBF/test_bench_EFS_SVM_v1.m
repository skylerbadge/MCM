clc;
clear all;

for dataset=[9]
    
    
    filename= sprintf('%d.mat',dataset);
    folds=sprintf('%dfold.mat',dataset);
    load(strcat('data_folds/',filename));
    load(strcat('data_folds/',folds));
    disp(filename)
    X=x;
    Y=y;
    clearvars('x');
    clearvars('y');
    nfolds=5;
    m=size(X,1);%size of training data
    %hyperparameters to be initialized here
    %kernel_type: 1:=RBF, 2:=poly, 3:=linear
    degree=[1e-04,1e-03,1e-02,1e-01,1,10,100];
    beta=[1e-04,1e-03,1e-02,1e-01,1,10,100];
    result=[];
    %similarly you can define beta here for RBF kernel and C for cost
    maxacc=0;
    type=1;%linear kernel
%     type=1%RBF kernel
    for k=1:length(beta)
        for z=1:length(degree)
            t1=[];
            t2=[];
            t3=[];
            t4=[];
            
            nf=0;
            for i=1:nfolds
                xTrain=[];
                yTrain=[];
                xTest=[];
                yTest=[];
                
                test = (indices == i);
                train = ~test;
                xTrain=X(train,:);
                yTrain=Y(train,:);
                xTest=X(test,:);
                yTest=Y(test,:);
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
                
                %add your own code here
                tic
                
                %for MCM_dual
                yTrain(yTrain==-1)=2;
                yTest(yTest==-1)=2;
                numClasses=length(unique(yTrain));
                
                [model_alpha,model_indices,trainAcc,model_exit ] = ovo_efs_svm_v1(xTrain,yTrain,beta(k),degree(z),numClasses,type);
                [ yt2] = get_prediction_ovo(xTrain,yTrain,xTest,model_alpha,model_indices,numClasses,beta(k),type);
                testAcc=sum(yt2==yTest)*100/size(yTest,1);
                
                
                
                
                %                     [trainAcc,testAcc,q,nsv] = lsMCMkernel_quadprog_rbf_L1_min(xTrain,yTrain,xTest,yTest,beta(k),degree(z),kernel_type);
                time=toc;
                
                weights=model_alpha(1:end-1,:);
                weights=weights(:);
                nsv=(size(weights,1)-sum(weights(:,1)==0))/(numClasses*(numClasses-1)/2);
                
                t1=[t1;trainAcc(1,1)];
                t2=[t2;testAcc(1,1)];
                t3=[t3;nsv];
                t4=[t4;time];
                %             fprintf('trainAcc = %0.2f, testAcc = %0.2f \n',trainAcc,testAcc);
                nf=nf+1;
                if((testAcc(1,1)< (maxacc/1.1)) || (testAcc(1,1)<60))
                    break;
                end
            end
            if(mean(t2)>maxacc)
                maxacc=mean(t2);
            end
            avg1=mean(t1);
            avg2=mean(t2);
            avg3=mean(t3);
            avg4=mean(t4);
            std1=std(t1);
            std2=std(t2);
            std3=std(t3);
            std4=std(t4);
            r=[avg1 std1 avg2 std2 avg3 std3 avg4 std4 beta(k) degree(z) nf];
            result=[result;r];
        end
        
    end
    [val,idx]=max(result(:,2));
    result=[result;result(idx,:)];
    result=full(result);
    result(end,:)
%     str1=sprintf('EFS_SVM_v2/result_dataset%d.csv',dataset);
%     save(str1,'result','-ascii');
end

