clc;clearvars;
rng default;
  
statfile = 'UCI_QMCM_Ker_RBF.txt';
ker_type = 'rbf';
f1 = fopen(statfile, 'at');
  
dlist=[1;2;4;5;7;8;9;11;13;15;16;17;18;19;20;21;23;24;25;26;27;28];
dlist=sort(dlist);
numPartition=5;
 
% add LIBSVM to path
addpath(genpath('/home/sumit/libsvm-3.21'));
 
for p=1:length(dlist)
dataset=dlist(p);
% for dataset=23:28
      
    fprintf(1,'----------Working on dataset %d-----------',dataset);
%     filename= sprintf('%d.mat',dataset);
%     folds=sprintf('%dfold.mat',dataset);
    load(strcat('/Users/cdac/Box Sync/sumit/unsupervised-test/semi-supervised/code/Folds/',filename));
    load(strcat('/Users/cdac/Box Sync/sumit/unsupervised-test/semi-supervised/code/Folds/',folds));
     
%     Server
    load(strcat('../datasets/Folds/',filename));
    load(strcat('../datasets/Folds/',folds));
     
    fprintf(f1,'\n --- Dataset: %d --- \n',dataset);
    X=x;
    Y=y;
    m=size(X,1);%size of training data
    nfolds=5;
    myfoldtime=zeros(nfolds,1);
    myfoldacc=zeros(nfolds,1);
    myfoldlibsvmacc=zeros(nfolds,1);
      
    for i=1:nfolds
           
        xTrain=[];
        yTrain=[];
        xTest=[];
        yTest=[];
          
        test = (indices == i);
        train = ~test;
        for j=1:m
            if(train(j)==1)
                xTrain=[xTrain;X(j,:)];
                yTrain=[yTrain;Y(j,:)];
            end
            if(test(j)==1)
                xTest=[xTest;X(j,:)];
                yTest=[yTest;Y(j,:)];
            end
        end
         
%         % Perform PCA
%         [coeff, score, latent, tsquared, explained]=pca(xTrain);
%         [ num_evs ] = selectBestEigenvalues( explained );
%         xTrain=score(:,1:num_evs);
%         proj_matrix=coeff(:,1:num_evs);
%         xTest=xTest*proj_matrix;
         
%          % Projection to random orthonormal basis
%          Dim=size(xTrain,2);
%          rand_proj_matrix=rand(Dim,Dim);
%          proj_matrix_orthogonalized=orth(rand_proj_matrix);
%          Dim_selected=round(Dim/2);
%          proj_matrix_orthogonalized=proj_matrix_orthogonalized(:,1:Dim_selected);
%          xTrain=xTrain*proj_matrix_orthogonalized;
%          xTest=xTest*proj_matrix_orthogonalized;
        %% Code for Q-MCM kernel Classifier
         
        [ bestC, bestD, bestKerPara, epsParam ] = Tune_qMCM_kernel( xTrain, yTrain, ker_type );
        [ lambda,b ] = qmcm_kernel( xTrain, yTrain, bestC, bestD, epsParam, ker_type, bestKerPara )
        [ accuracy, yPred ] = computePredictions( xTrain,xTest,yTest,lambda,b,ker_type,bestKerPara );
             
        myfoldacc(i)=accuracy;
         
        [ Cbest, gammaBest] = TuneSVM( xTrain, yTrain,ker_type );
        model=svmtrain(yTrain,xTrain,['-q -t 2 -c ',num2str(Cbest),' -g ', num2str(gammaBest)]);
        [predVal,accuracy,decVals]=svmpredict(yTest,xTest,model,'-q');
        myfoldlibsvmacc(i)=accuracy(1);
         
 
        fprintf(f1, '\n Acc Our: %.2f \t SVM: %.2f ', myfoldacc(i), myfoldlibsvmacc(i));
    end
    fprintf(f1, '\nAvg : Acc Our: %.2f +/- %.2f, SVM:  %.2f +/- %.2f\n',mean(myfoldacc), std(myfoldacc), mean(myfoldlibsvmacc), std(myfoldlibsvmacc));
end
  
fclose(f1);