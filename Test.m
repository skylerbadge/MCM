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

kerType={'poly','poly'};
alpha=[0.25;0.25];
kerPara=[4;5];

kerTypeMCM=kerType{1};
kerParaMCM=kerPara(1);
Cparam=0.0085;



% Compute MCM accuracy
[ lambda,b,h ] = mcm_linear_efs( xTrain, yTrain, kerTypeMCM, kerParaMCM, Cparam );
[ pred,acc_mcm ] = mcmPredict( xTrain,xTest,yTest,kerTypeMCM,kerParaMCM,lambda,b );
acc_mcm