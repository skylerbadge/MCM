function [ Cbest,  bestKerPara] = tuneMCM_v2( xTrain, yTrain, kerTypeMCM , cInit , kerParams )
    
    [N, D]=size(xTrain);
    split_pt=round(0.8*N);
    xTrain1=xTrain(1:split_pt,:);
    yTrain1=yTrain(1:split_pt,:);
    xValid=xTrain(split_pt+1:end,:);
    yValid=yTrain(split_pt+1:end,:);
    bestAcc=0;
    bestKerPara=0;
    
    for i=1:length(kerParams)
        
        Ctest=cInit;
        TrainAcc = 50;
        TestAcc = 0;

        ctr = 15; 
        kerParaMCM=kerParams(i);
        
        while(abs(TrainAcc-TestAcc)>4 && TestAcc<TrainAcc && ctr>0)
            try
                [ lambda,b,h ] = mcm_linear_efs( xTrain1, yTrain1, kerTypeMCM, kerParaMCM, Ctest );
            catch
                lambda = rand(size(xTrain1,1),1);
                b = rand;
            end
            
            if (TrainAcc>TestAcc)
                Ctest=Ctest/4;
            else
                Ctest=Ctest*4;
            end
            
            [ pred , TrainAcc ] = mcmPredict( xTrain1,xTrain1,yTrain1,kerTypeMCM,kerParaMCM,lambda,b );
            [ pred , TestAcc ] = mcmPredict( xTrain1,xValid,yValid,kerTypeMCM,kerParaMCM,lambda,b );
            
            ctr = ctr-1;

            if(TestAcc > bestAcc)
                bestAcc = TestAcc;
                Cbest=Ctest;
                bestKerPara=kerParaMCM;
            end 
        end
    end
    
    fprintf(2,'Tuning done: C=%.2f, kerParaMCM=%.2f, bestTrainAcc=%.2f \n', Cbest, bestKerPara, bestAcc);
    

end