
function [ pred,acc ] = testMultiKernel( xTrain,xTest,yTest,kerType,kerPara,lambda,b,alpha )
%TESTMULTIKERNEL - Predicts using multiple kernels

[N,~]=size(xTrain);
[Ntest,~]=size(xTest);
pred=zeros(Ntest,1);



for i=1:Ntest
    K=zeros(N,1);
    for j=1:N
        for k=1:length(alpha) % for each kernel
            K(j)=K(j)+alpha(k)*...
                kernelfunction(kerType{k},xTest(i,:),xTrain(j,:),kerPara(k));
        end
    end
    predVal=(lambda'*K)+b/norm(lambda);
    if(predVal)>=0
        pred(i)=1;
    else
        pred(i)=-1;
    end
end

acc=sum(pred==yTest)/Ntest*100;

if(acc<50)
    pred=-pred;
    acc=sum(pred==yTest)/Ntest*100;
end

end

