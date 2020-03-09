function [ Cbest ] = tuneMCMmultker( xTrain, yTrain, kerTypeMCM , cParams , gamma , alpha )
    
    [N, D]=size(xTrain);
    split_pt=round(0.8*N);
    xTrain1=xTrain(1:split_pt,:);
    yTrain1=yTrain(1:split_pt,:);
    xValid=xTrain(split_pt+1:end,:);
    yValid=yTrain(split_pt+1:end,:);
    bestAcc=0;
    Cbest=0;
    tempalpha = alpha;
    for i=1:length(cParams)
    
        Ctest=cParams(i);
        
        iterMax=10;
        new_acc=1;
        old_acc=0;
        iter=1;
        
        alpha = tempalpha;

        while (iter<=iterMax && new_acc>old_acc)
            try
                [ lambda,b,h ] = mcm_linear_efs_multker( xTrain1, yTrain1, kerTypeMCM , gamma, Ctest, alpha );
            catch
                lambda = rand(size(xTrain,1),1);
                b = rand;
            end

            try
                [ alpha ] = mcm_ker_mult_v2( xTrain1, yTrain1, kerTypeMCM, gamma , Ctest, lambda);    
            catch
                alpha = [0.2*rand;0.2*rand;0.2*rand;0.2*rand];
                alpha = [alpha ; 1-sum(alpha)];
                %assigning random and setting the last one such that the
                %sum is 1
            end

%             fprintf(2,'\n***********************************************************\n');
            [ pred,acc ] = testMultiKernel( xTrain1,xValid,yValid,kerTypeMCM, gamma ,lambda,b,alpha );

%             fprintf(2,'Accuracy changed from %.3f to %.3f in iteration %d \n',old_acc,acc,iter);
%             alpha'

            % Swap accuracy variable
            temp=new_acc;
            new_acc=acc;
            old_acc=temp;

            % Increment iteration counter
            iter=iter+1;
        end
        if(acc >= bestAcc)
                bestAcc = acc;
                Cbest = Ctest;
        end
    end
    
    fprintf(2,'Tuning done: C=%.2f , bestTrainAcc=%.2f \n', Cbest, bestAcc);
    