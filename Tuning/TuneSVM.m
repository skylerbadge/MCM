function [ Cbest,  bestKerPara] = TuneSVM( trainData, trainLabels, ker_type )

% Initializations
C1_range=[0.1;1;20;100];
if (strcmp(ker_type,'rbf'))
    kerParamRange=[0.001;0.01;0.1;1;10;100];
else
    kerParamRange=[3;4;5];
end

% Separate validation set
[N, D]=size(trainData);
split_pt=round(0.8*N);
xTrain=trainData(1:split_pt,:);
yTrain=trainLabels(1:split_pt,:);
xTest=trainData(split_pt+1:end,:);
yTest=trainLabels(split_pt+1:end,:);
bestAcc=0;
Cbest=0;bestKerPara=0;

% Tune
for i=1:length(C1_range)
    
    for k=1:length(kerParamRange)
        C1=C1_range(i);
        
        ker_para=kerParamRange(k);
        if(strcmp(ker_type,'rbf'))
            model=svmtrain(yTrain,xTrain,['-t 2 -c ',num2str(C1),' -g ', num2str(ker_para)]);
        else
            model=svmtrain(yTrain,xTrain,['-t 1 -c ',num2str(C1),' -d ', num2str(ker_para)]);
        end
        [predVal,accuracy,decVals]=svmpredict(yTest,xTest,model);
        if(accuracy(1)>=bestAcc)
            bestAcc=accuracy(1);
            Cbest=C1;
            bestKerPara=ker_para;
        end
    end
    
end

fprintf(2,'Tuning done: C1=%.2f, ker_para=%.2f, bestTrainAcc=%.2f \n', Cbest, bestKerPara, bestAcc);


end

