function [ alpha ] = mcm_ker_mult_v2( xTrain, yTrain, kerType, kerPara, Cparam, lambda )
%MCM_KER_MULT  - solves for the kernel multipliers using MCM linear EFS
% xTrain - training data
% yTrain - training labels
% kerType - array of kernel types
% kerPara - array of kernel parameters
% Cparam - hyperparameter C
% lambda - hyperplane weight vector in kernel space
% b - hyperplane bias
% h - MCM VC dimension term

[N,D]=size(xTrain);
numKernels=length(kerType);

% Initialization
X0=[rand(numKernels,1);rand(N,1);rand;rand]; %alpha,q,h,b

% Linear term in objective
f=[zeros(numKernels,1);Cparam*ones(N,1);1;0];

% Setup inequality constraints

% y(lambda \sum \alpha_i K_i + b) + q >= 1
A_ineq1=[];
b_ineq1=[-ones(N,1)];
for i=1:numKernels
    A_ineq1=[A_ineq1,-computeKernelMatrix(kerType{i},kerPara(i),xTrain)*(yTrain.*lambda)];
end
A_ineq1=[A_ineq1,-eye(N),zeros(N,1),-yTrain];

% h >= y(lambda \sum \alpha_i K_i + b) + q
A_ineq2=[];
b_ineq2=[ones(N,1)];
for i=1:numKernels
    A_ineq2=[A_ineq2,computeKernelMatrix(kerType{i},kerPara(i),xTrain)*(yTrain.*lambda)];
end
A_ineq2=[A_ineq2,eye(N),-ones(N,1),yTrain];

A_ineq=[A_ineq1;A_ineq2];
b_ineq=[b_ineq1;b_ineq2];

% Setup equality constraints
A_eq=[ones(1,numKernels),zeros(1,N),0,0]; %\sum \alpha = 1
b_eq=[1];

% Setup bounds
lb=[zeros(numKernels,1);zeros(N,1);1;-inf];
ub=[ones(numKernels,1);inf(N,1);inf;inf];

options = optimoptions('linprog','Algorithm','dual-simplex'); 

% Solve LPP
[X,fVal,EXITFLAG]=linprog(f,A_ineq,b_ineq,A_eq,b_eq,lb,ub,options);

if (EXITFLAG>=0)
    alpha=X(1:numKernels);
else
    alpha=zeros(numKernels,1);
end


end

function [K]=computeKernelMatrix(kerType,kerPara,xTrain)
[Ns,~]=size(xTrain);
K=zeros(Ns,Ns);
for i=1:Ns
    for j=1:Ns
        K(i,j)=kernelfunction(kerType,xTrain(i,:),xTrain(j,:),kerPara);
    end
end
end

