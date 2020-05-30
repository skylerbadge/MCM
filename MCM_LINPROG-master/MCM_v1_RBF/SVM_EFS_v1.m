function [ alpha,b_offset,exit ] = SVM_EFS_v1( xTrain,yTrain,C1,C2,type )
%SVM_EFS_V1 Summary of this function goes here
%   Detailed explanation goes here
% min |lambda|+C*summation(q_i)
% y_i(\summation_j \lambda_j\K(x_i,x_j)+b)+q_i>=1
% q_i>=0
% C1=kernel parameter (beta)
% C2=C in SVM
N=size(xTrain,1);
M=size(xTrain,2);
f=[zeros(N,1);zeros(1,1);C2*ones(N,1);ones(1,1)];%lambda,b,q1,h
A=zeros(2*N,2*N+2);
b=zeros(2*N,1);

%-y_i(\summation_j(\lambda_i K_ij)+b)-q_i<=-1
for i=1:N
    for j=1:N
        A(i,j)=-yTrain(i)*kernel(xTrain(i,:),xTrain(j,:),C1,type);%\summation \lambda_i\K(x_i,x_j)
    end
    A(i,N+1)=-yTrain(i);%b
    A(i,N+1+i)=-1;%q_i
    b(i,1)=-1;
end
%y_i(\summation_j(\lambda_i K_ij)+b)+q_i-h<=0
for i=1:N
    for j=1:N
        A(i+N,j)=-A(i,j);%\summation \lambda_i\K(x_i,x_j)
    end
    A(i+N,N+1)=-A(i,N+1);%b
    A(i+N,N+1+i)=1;%q_i
    A(i+N,2*N+2)=-1;%h
    b(i+N,1)=0;
end
options=optimoptions('linprog','Algorithm','dual-simplex','Display','off');
lb =[-inf*ones(N,1);-inf*ones(1,1);zeros(N,1);ones(1,1)];
ub = [inf*ones(N,1);inf*ones(1,1);inf*ones(N,1);inf*ones(1,1)];


[X, fval, exitflag]  = linprog(f,A,b,[],[], lb,ub, [], options);

if(exitflag~=1)
    alpha=zeros(N,1);
    b_offset=zeros(1,1);
    exit=0;
    disp(exitflag);
    return;
end
exit=1;

alpha=X(1:N,1);
b_offset=X(N+1,1);
max_alpha=max(abs(alpha));
alpha(abs(alpha)<1e-04*max_alpha)=0;

end

