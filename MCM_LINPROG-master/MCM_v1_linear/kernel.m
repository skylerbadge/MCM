function [ k ] = kernel(x1,x2,beta,type)
%KERNEL Summary of this function goes here
%   Detailed explanation goes here
if(type==1)%RBF kernel
    k=exp(-beta*(norm(x1-x2))^2);
elseif(type==2)
   k=x1*x2'; 
else
    k=(x1*x2')*beta;
end
end

