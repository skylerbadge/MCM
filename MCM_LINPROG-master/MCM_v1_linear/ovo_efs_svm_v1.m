function [ model_alpha,model_indices,accTrain,model_exit] = ovo_efs_svm_v1(traindata,trainlabels,C1,C2,numClasses,type)
%this assumes that the data is already zero mean and unit variance

% traindata = samples X (features+1)
% trainlabels = samples X 1
% trainlabels should be from 1:numClasses
% we form numClasses x(numClasses - 1) classifiers
% model contains the weights of the classifier
model_alpha=zeros(size(traindata,1)+1,numClasses*(numClasses-1)/2);
model_indices=zeros(numClasses*(numClasses-1)/2,2);
% model_accuracy=zeros(1,numClasses*(numClasses-1)/2);
model_exit=zeros(1,numClasses*(numClasses-1)/2);
model_idx=0;

% indices=zeros(1,2);
for i=1:numClasses
    for j=i+1:numClasses
        %class1 +ve class
        %class2 -ve class
        class1_idx=trainlabels==i;
        class2_idx=trainlabels==j;
        
        xTrain=[traindata(class1_idx,:);traindata(class2_idx,:)];
        yTrain=[ones(sum(class1_idx),1);-ones(sum(class2_idx),1)];
        
        [alpha,b,exit ] = SVM_EFS_v1( xTrain,yTrain,C1,C2,type);
        
        %clearvars('xTrain');
        %clearvars('yTrain');
        %         fprintf('Accuracy= %0.2f\n',trainAcc);
        
        indices=[i j];
        model_idx=model_idx+1;
        model_alpha(:,model_idx)=[alpha;b];
        %         model_accuracy(:,model_idx)=trainAcc;
        
        model_exit(:,model_idx)=exit;
        model_indices(model_idx,:)=indices;
        
        
    end
end
[ yt1 ] = get_prediction_ovo(traindata,trainlabels,traindata,model_alpha,model_indices,numClasses,C1,type);
accTrain=sum(yt1==trainlabels)*100/size(trainlabels,1);
end

