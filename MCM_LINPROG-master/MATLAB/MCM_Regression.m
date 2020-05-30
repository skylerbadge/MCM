function [yPred, testmse, nsv]= MCM_Regression(xTrain, yTrain, xTest, yTest, C, beta, epsilon)

% Jayadeva, July 2017
% This function implements the MCM Regressor (MCMR) via classification. 
% The code is based on the algorithm in the following paper:
% Jayadeva, Chandra, Suresh, Sanjit S. Batra, and Siddarth Sabharwal. 
% "Learning a hyperplane regressor through a tight bound on the VC dimension." 
% Neurocomputing 171 (2016): pp. 1610-1616.
% 
% Inputs:
% xTrain: Training dataset in the form train_samplesXfeatures
% yTrain: Training labels in the form train_samplesX1
% xTest: Testing dataset in the form test_samplesX1
% yTest: Testing labels in the form test_samplesX1
% C: Parameter for weight on error terms in MCMR Objective function
% beta: Kernel parameter for MCMR (kernel width in case of RBF Kernel
% epsilon: Parameter for converting regression problem to classification
% 
% Outputs:
% yPred: Predictions on test data computed using MCMR (test_samplesX1)
% testmse: Mean Squared Error (MSE) on test set predictions
% nsv: Number of support vectors computed by MCMR model on training data
% 
% Notes:
% 1. The yTest input is used to compute MSE on the MCMR predictions, in
% case it is not available, please pass rand(test_samples,1) as a dummy
% vector of random test sample predictions, of size test_samplesX1
% 2. The kernel function to be used is mentioned in lines 37/39 of this
% function, please choose the suitable kernel function to be used.
% 3. The number of support vectors are computed by thresholding the
% solution vector lambda, the threshold can be set on line 42.



% Kernel definitions
% RBF kernel
Kernel = @(x,y,beta) exp(-beta * norm(x-y)^2);
% Linear kernel
% Kernel = @(x,y,beta) ( (x*y'));

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



N=size(xTrain,1);
D=size(xTrain,2);


% Normalizing values of training and test dataset

% Normalize x_values
m = mean(xTrain);
stdev = std(xTrain);
for d=1:D
    if(stdev(d)~=0)
        xTrain(:,d) = (xTrain(:,d) - m(d))/stdev(d);
        xTest(:,d) = (xTest(:, d) - m(d))/stdev(d);
    else
        xTrain(:,d) = (xTrain(:,d) - m(d));
        xTest(:,d) = (xTest(:, d) - m(d));
    end
end


%[lambda   ;b         ; Q+       ; Q-       ;H         ;P  ]
X = [randn(N,1);randn(1,1);randn(N,1);randn(N,1);randn(1,1);randn(1,1)];       %[lambda,b,q+,q-,H,P]
f = [zeros(N,1);zeros(1,1);C*ones(N,1);C*ones(N,1);ones(1,1);zeros(1,1)];

eps_1 = yTrain + epsilon*(ones(N,1));%(Yi + epsilon)
eps_2 = yTrain - epsilon*(ones(N,1)) ;% ( Yi - epsilon)


A = zeros(4*N, 3*N + 3);
b = zeros(4*N, 1);


%  implementing (lambda)*K(xi,xj) +b +p(Yi + epsilon) - H <= 0
for i = 1:N
    for j = 1:N
        atemp = Kernel(xTrain(i, :), xTrain(j, :), beta);
        A(i, j) = atemp;% lambda
    end
    A(i,  N+1) = 1; %b
    A(i, N+1 + i) = 0;%q+
    A(i,2*N+1 + i) = 0;%q-
    A(i,3*N + 2) = -1;%H
    A(i,3*N + 3) = eps_1(i);%P
    b(i,1) = 0;
end

%  implementing -(lambda)*K(xi,xj) -b -p(Yi - epsilon) -H <= 0

offset = N;
for i = 1:N
    for j = 1:N
        atemp = Kernel(xTrain(i, :), xTrain(j, :), beta);
        A( offset + i, j) = -atemp;%lambda
    end
    A(offset + i,N+1) = -1; %b
    A(offset + i,N+1 + i) = 0;%q+
    A(offset + i,2*N+1 + i) = 0;%q-
    A(offset + i,3*N + 2) = -1;%H
    A(offset + i,3*N + 3) = -1*eps_2(i);%P
    b(offset + i,1) = 0;
end

%  implementing -(lambda)*K(xi,xj) - b - p(y + epsilon) - q+  <= -1


offset = 2*N;
for i = 1:N
    for j = 1:N
        atemp = Kernel(xTrain(i, :), xTrain(j, :), beta);
        A( offset + i, j) = -atemp;%lambda
    end
    A(offset + i,N+1) = -1; %b
    A(offset + i,N+1 + i) = -1;%q+
    A(offset + i,2*N+1 + i) = 0;%q-
    A(offset + i,3*N + 2) = 0;%H
    A(offset + i,3*N + 3) = -1*eps_1(i);%P
    b(offset + i,1) = -1;
end

%  implementing (lambda)*K(xi,xj) + b + p(y - epsilon) - q-  <= -1

offset = 3*N;
for i = 1:N
    for j = 1:N
        atemp = Kernel(xTrain(i, :), xTrain(j, :), beta);
        A( offset + i, j) = atemp;%lambda
    end
    A(offset + i,N+1) =  1; %b
    A(offset + i,N+1 + i) = 0;%q+
    A(offset + i,2*N+1 + i) = -1;%q-
    A(offset + i,3*N + 2) = 0;%H
    A(offset + i,3*N + 3) = eps_2(i);%P
    b(offset + i,1) = -1;
end





Aeq = [];
beq = [];

lb = [-inf*ones(N,1);-inf*ones(1,1);0*ones(N,1);0*ones(N,1);0;-inf];
ub = [ inf*ones(N,1); inf*ones(1,1);inf*ones(N,1);inf*ones(N,1);inf;inf];
options = optimoptions('linprog','display','final', 'Algorithm', 'dual-simplex');

[X,~,exitflag] = linprog(f,A,b,Aeq,beq,lb,ub,X,options);

if(exitflag ~= 1)
    fprintf(2,strcat('\n****************************************************',...
        '\nPlease choose suitable values of C and beta, optimization routine did not converge \n',...
        '\n****************************************************\n'));
    return;
end

lambda = X(1:N,:);

P = full(X(3*N+3,:));
bias = X(N+1,:);
ntest = size(xTest,1);
yPred = zeros( size(xTest,1),1);


if (P==0)
    fprintf(2, strcat('\n****************************************************',...
        '\nERROR: Model not solved to optimality, please choose different values of parameters!',...
        '\n****************************************************\n'));
    return;
end

% Compute predictions on test data
for i = 1:ntest
    sum = bias;
    for j = 1:N
        sum = sum + (lambda(j)) * Kernel(xTrain(j, :), xTest(i, :), beta);
    end
    yPred(i) = (-1/P)*(sum);
end
testmse = norm(yPred-yTest)^2/length(yTest);


% Compute number of support vectors
nsv = 0;
for i = 1: N
    
    if(abs(lambda(i))>svThreshold)
        nsv = nsv + 1;
    end
end


% Display results

fprintf(1, strcat('\n****************************************************',...
    '\nMCMR Training and prediction completed!\n',...
    '\nTest Data Mean Squared Error = %.2f\nNo. of support vectors=%.2f\n',...
    '\n****************************************************\n'),...
    testmse,nsv);

end



