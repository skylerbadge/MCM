function [ yt1 ] = get_prediction_ovo(traindata,trainlabels,xTest,model_alpha,model_indices,numClasses,beta,type)
%GET_PREDICTION_OVO Summary of this function goes here
%   Detailed explanation goes here

% xTrain is of size samples x (features+1)
N=size(xTest,1);
yt1=zeros(N,1);
for i=1:N
    yval=zeros(numClasses*(numClasses-1)/2,1);
    %     data=xTrain(i,:);
    %     phix=xTrain(i,:);
    for n=1:numClasses*(numClasses-1)/2
        alpha=model_alpha(1:end-1,n);
        b_offset=model_alpha(end,n);
        indices=model_indices(n,:);
        sum1=0;
        class1_idx=(trainlabels==indices(1,1));
        class2_idx=(trainlabels==indices(1,2));
        xTrain=[traindata(class1_idx,:);traindata(class2_idx,:)];
        for j=1:size(alpha,1)
            if(alpha(j)~=0)
                sum1=sum1+alpha(j)*kernel(xTrain(j,:),xTest(i,:),beta,type);
            end
            
        end
        output=sum1+b_offset;
        %         output=phix*W;
        
        if(output>0)
            yval(n)=indices(1,1);
        else
            yval(n)=indices(1,2);
        end
    end
    yt1(i)=mode(yval);
end


end

