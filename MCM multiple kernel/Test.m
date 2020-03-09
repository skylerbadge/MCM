clc;clearvars;close all;
rng default;

% % Iris
% load('fisheriris.mat');
% xTrain=meas([1:30,51:80],:);
% yTrain=[ones(30,1);-ones(30,1)];
% xTest=meas([31:50,81:100],:);
% yTest=[ones(20,1);-ones(20,1)];

%Ionosphere
load('ionosphere.mat');
yTrain=[];
for i=1:length(Y)
    if (Y{i}=='b')
        yTrain=[yTrain;1];
    else
        yTrain=[yTrain;-1];
    end
end
classA=X(yTrain==1,:);
classB=X(yTrain==-1,:);
nA=round(0.8*size(classA,1));
nB=round(0.8*size(classB,1));
xTrain=[classA(1:nA,:);classB(1:nB,:)];
xTest=[classA(nA+1:end,:);classB(nB+1:end,:)];
yTrain=[ones(nA,1);-ones(nB,1)];
yTest=[ones(size(classA,1)-nA,1);-ones(size(classB,1)-nB,1)];

kerType={'rbf','rbf','rbf','rbf'};
alpha=[0.25;0.25;0.25;0.25];
kerPara=[0.1,1,0.001,0.01];

kerTypeMCM=kerType{1};
kerParaMCM=kerPara(1);
Cparam=85;



iterMax=10;
new_acc=1;
old_acc=0;
iter=1;
tempalpha = alpha
while (iter<=iterMax && new_acc>old_acc)
    
[ lambda,b,h ] = mcm_linear_efs_multker( xTrain, yTrain, kerType, kerPara, Cparam, tempalpha );

[ tempalpha ] = mcm_ker_mult_v2( xTrain, yTrain, kerType, kerPara, Cparam, lambda);

fprintf(2,'\n***********************************************************\n');
[ pred,acc ] = testMultiKernel( xTrain,xTrain,yTrain,kerType,kerPara,lambda,b,tempalpha );

fprintf(2,'Accuracy changed from %.3f to %.3f in iteration %d \n',old_acc,acc,iter);
tempalpha'


% Swap accuracy variable
% temp=new_acc;
% new_acc=acc;
% old_acc=temp;

old_acc = new_acc;
new_acc = acc;

if (new_acc > old_acc)
    alpha = tempalpha
end

% Increment iteration counter
iter=iter+1;
end

[ pred,acc_multker ] = testMultiKernel( xTrain,xTest,yTest,kerType,kerPara,lambda,b,alpha );
acc_multker



% Compute MCM accuracy
[ lambda,b,h ] = mcm_linear_efs( xTrain, yTrain, kerTypeMCM, kerParaMCM, Cparam );
[ pred,acc_mcm ] = mcmPredict( xTrain,xTest,yTest,kerTypeMCM,kerParaMCM,lambda,b );
acc_mcm