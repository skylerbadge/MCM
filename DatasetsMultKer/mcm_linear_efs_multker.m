function [ lambda,b,h ] = mcm_linear_efs_multker( xTrain, yTrain, kerType, kerPara, Cparam, alpha )
%MCM_LINEAR_EFS -  solves MCM for multiple weighted kernels
% Linear MCM in EFS with multiple kernels
% xTrain - training data
% yTrain - training labels
% kerType - array of kernel types
% kerPara - array of kernel parameters
% Cparam - hyperparameter C
% alpha - array of kernel multiplying coefficients

[N,D]=size(xTrain);

% Solution variables
X0=[rand(N,1);rand;rand;rand(N,1)]; % w,b,h,q

% Compute kernel matrix
K=zeros(N,N);

for i=1:N
    for j=1:N
        for k=1:length(alpha) % for each kernel
            K(i,j)=K(i,j)+alpha(k)*...
                kernelfunction(kerType{k},xTrain(i,:),xTrain(j,:),kerPara(k));
        end
    end
end

% Linear term in objective funcion
f = [zeros(N,1);0;1;Cparam*ones(N,1)]; % h + C \sum q

% Equality constraints
A_eq=[];
b_eq=[];

% Inequality constraints
A_ineq=[-diag(yTrain)*K,-yTrain,zeros(N,1),-eye(N);... % y(lambda K + b) + q >= 1
    diag(yTrain)*K,yTrain,-ones(N,1),eye(N)]; % h >= y(lambda K + b) + q
b_ineq=[-ones(N,1);zeros(N,1)];

% Bounds
lb=[-inf(N,1);-inf;1;zeros(N,1)];
ub=[inf(N,1);inf;inf;inf(N,1)];

options = optimoptions('linprog','Algorithm','dual-simplex','MaxTime',120); 

% Solve LPP
[X,fVal,EXITFLAG]=linprog(f,A_ineq,b_ineq,A_eq,b_eq,lb,ub,options);

if (EXITFLAG>=0)
    lambda=X(1:N);
    b=X(N+1);
    h=X(N+2);
else
    lambda=zeros(N,1);
    b=0;
    h=0;
end
end

