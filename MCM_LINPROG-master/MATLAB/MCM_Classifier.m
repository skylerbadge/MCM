function [ yPred, testAcc, nsv ] = MCM_Classifier( xTrain, yTrain, xTest, yTest, C, beta )

% MCM_Classifier 
% Jayadeva, July 2017
% This function implements the MCM for classification. 
% The code is based on the algorithm in the following paper:
% Jayadeva. "Learning a hyperplane classifier by minimizing an 
% exact bound on the VC dimension." Neurocomputing 149 (2015): 683-689.
% 
% Inputs:
% xTrain: Training dataset in the form train_samplesXfeatures
% yTrain: Training labels in the form train_samplesX1 (should be +1/-1)
% xTest: Testing dataset in the form test_samplesX1
% yTest: Testing labels in the form test_samplesX1 (should be +1/-1)
% C: Parameter for weight on error terms in MCM Objective function
% beta: Kernel parameter for MCM (kernel width in case of RBF Kernel
% 
% Outputs:
% yPred: Predictions on test data computed using MCM (test_samplesX1)
% testAcc: %age accuracy on test set predictions
% nsv: Number of support vectors computed by MCM model on training data
% 
% Notes:
% 1. The yTest input is used to compute accuracy on the MCM predictions, in
% case it is not available, please pass ones(test_samples,1) as a dummy
% vector of random test sample predictions, of size test_samplesX1
% 2. The kernel function to be used is mentioned in lines 37/39 of this
% function, please choose the suitable kernel function to be used.
% 3. The number of support vectors are computed by thresholding the
% solution vector lambda, the threshold can be set on line 42.




% Define Kernel
% Linear
% Kernel = @(x,y) ( (x*y'));
% RBF
Kernel = @(x,y) exp(-beta * norm(x-y)^2);

% Threshold to compute number of support vectors
svThreshold=1e-3;



if (size(xTrain,2)~=size(xTest,2))
     fprintf(2, strcat('\n****************************************************',...
        '\nERROR: Number of features of training and test data are not same!',...
        '\n****************************************************\n'));
    return;
end

if (size(xTrain,1)~=size(yTrain,1))
     fprintf(2, strcat('\n****************************************************',...
        '\nERROR: Number of training samples and training labels are not same!',...
        '\n****************************************************\n'));
    return;
end

if (size(xTest,1)~=size(yTest,1))
     fprintf(2, strcat('\n****************************************************',...
        '\nERROR: Number of testing samples and testing labels are not same!',...
        '\n****************************************************\n'));
    return;
end


N = size(xTrain,1);
D = size(xTrain,2);
Ntest=size(xTest,1);


%solve linear program
%   [    lambda,         b,          q,         h]
X = [randn(N,1);randn(1,1); randn(N,1);randn(1,1);];
f = [zeros(N,1);zeros(1,1);C*ones(N,1);         1;];

LM = zeros(N,N);
for i=1:N
    for j=1:N
        LM(i,j) = yTrain(i) * Kernel(xTrain(i,:),xTrain(j,:));
    end
end

LX = zeros(D,N);
for i=1:D
    for j=1:N
        LX(i,j) = xTrain(j,i);
    end
end



%   [lambda,               b,          q,              h]
A = [ LM        ,     yTrain, zeros(N,N),   -1*ones(N,1);
    -LM        ,    -yTrain,  -eye(N,N),     zeros(N,1);];

B = [zeros(N,1);-1*ones(N,1);];


Aeq = [];
Beq = [];

%    [        lambda,      b,             q,     h]
lb = [-inf*ones(N,1);   -inf;    zeros(N,1);     0;];
ub = [ inf*ones(N,1);    inf; inf*ones(N,1);   inf;];

options = optimoptions('linprog','display','final', 'Algorithm', 'dual-simplex');

[X, fval, exitflag]  = linprog(f,A,B,Aeq,Beq, lb,ub,X,options);
lambda = X(1:N,:);
b = X(N + 1,:);
q = X(N+1 + 1:N+1 + N,:);
h = X(2*N+1 + 1,:);

yPred = zeros(Ntest, 1);

for i = 1:Ntest
    sumj = b;
    for j = 1:N
        sumj = sumj + lambda(j) * Kernel(xTrain(j, :), xTrain(i, :));
    end
    if (sumj>0)
        yPred(i) = 1;
    else
        yPred(i) = -1;
    end
end

testAcc = sum(yPred==yTest)/size(yTest,1) * 100;


nsv = 0;
for i = 1: N
    if (abs(lambda(i)) >svThreshold)
        nsv = nsv+1;
    end
end


% Display results

fprintf(1, strcat('\n****************************************************',...
    '\nMCM Training and prediction completed!\n',...
    '\nTest Data Accuracy (percentage) = %.2f\nNo. of support vectors=%.2f\n',...
    '\n****************************************************\n'),...
    testAcc,nsv);



end

