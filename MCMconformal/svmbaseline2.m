clc;
clear all;

for dataset=[4]%change the number from 1 to 22 you can put this in a loop like this
    
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
    gamma=[1e-04,1e-03,1e-02,1e-01,1,10,100];
    cpara=[1e-04,1e-03,1e-02,1e-01,1,10,100];
    result=[];
    %similarly you can define beta here for RBF kernel and C for cost
    maxacc=0;
    type=2;%linear kernel
%     type=1%RBF kernel
    for k=1:length(cpara)
        for z=1:length(gamma)
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
                
                
                model = svmtrain(yTrain, xTrain, sprintf('-t 3 -g %d -c %d',gamma(z),cpara(k)));
                [~, testAcc,~] = svmpredict(yTest, xTest, model);
                [~, trainAcc,~] = svmpredict(yTrain,xTrain, model);
                 
                time=toc;
                    
                nsv=model.totalSV;
                
                t1=[t1;trainAcc(1,1)];
                t2=[t2;testAcc(1,1)];
                t3=[t3;nsv];
                t4=[t4;time];
                
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
            r=[avg1 avg2 avg3 avg4 std1 std2 std3 std4 cpara(k) gamma(z) nf];
            result=[result;r];
        end
        
    end
    [val,idx]=max(result(:,2));
    result=[result;result(idx,:)];
    result=full(result); 
    
    xlswrite(strcat(int2str(dataset),'_result_baseline_svm_gs.xlsx'),result)
end