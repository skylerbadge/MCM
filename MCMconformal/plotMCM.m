clc;
clear all;

gamma=2.^[-10,-11,-12,-8,-9];

cParams = 2.^(-19:2:2);

kerTypeMCM ='rbf';

for dataset=[26]
    
    
    filename= sprintf('%d.mat',dataset);
    folds=sprintf('%dfold.mat',dataset);
    load(strcat('data_folds/',filename));
    load(strcat('data_folds/',folds));
    clc;
    disp(dataset);

    X=x;
    Y=y;

    s=size(X,1);

    for fold = [2]
        graphs(fold)=figure;
        dotsize = 20;
        colormap([1 0 .5;   % magenta
                   0 0 .8;   % blue
                   0 .6 0;   % dark green
                   .3 1 0]); % bright green

        xTrain0=[];
        yTrain=[];
        xTest0=[];
        yTest=[];


        test = (indices == fold);
        train = ~test;
        for j=1:s
            if(train(j)==1)
                xTrain0=[xTrain0;X(j,:)];
                yTrain=[yTrain;Y(j,:)];
            end
            if(test(j)==1)
                xTest0=[xTest0;X(j,:)];
                yTest=[yTest;Y(j,:)];
            end
        end
        %data preprocessing
        me=mean(xTrain0);
        std_dev=std(xTrain0);
        xTrain = zeros(size(xTrain0));
        xTest = zeros(size(xTest0));
        for n=1:size(xTrain,2)
            if(std_dev(n)~=0)
                xTrain(:,n)=(xTrain0(:,n)-me(n))./std_dev(n);
            else
                xTrain(:,n)=(xTrain0(:,n)-me(n));
            end
        end
        for n=1:size(xTest,2)
            if(std_dev(n)~=0)
                xTest(:,n)=(xTest0(:,n)-me(n))./std_dev(n);
            else
                xTest(:,n)=(xTest0(:,n)-me(n));
            end
        end
%         [coeff,score] = pca(xTrain);
% %         subplot(3,2,1);
%         scatter(score(:,1), score(:,2), dotsize, yTrain,'filled');
% %         title('Train Data embedding without any kernel optimization')

%         [coeff,score] = pca(xTest);
% %         subplot(3,2,2);
%         scatter(score(:,1), score(:,2), dotsize, -yTest,'filled');
% %         title('Test Data embedding without any kernel optimization')



        [Ctest,testKerPara] = tuneMCM( xTrain, yTrain , kerTypeMCM , cParams , gamma);
        try
            [ lambda,b,h ] = mcm_linear_efs( xTrain, yTrain, kerTypeMCM, testKerPara, Ctest );
        catch
            lambda = rand(size(xTrain,1),1);
            b = rand;
        end

        [N,D]=size(xTrain);
        % Compute kernel matrix
        K=zeros(N,N);
        for i=1:N
            for j=1:N
                K(i,j)=K(i,j)+...
                    kernelfunction(kerTypeMCM,xTrain(i,:),xTrain(j,:),testKerPara);
            end
        end
% 
%         [coeff,score] = pca(K);
% %         subplot(3,2,3);
%         scatter(score(:,1), score(:,2), dotsize, yTrain,'filled');
% %         title('Train Data embedding with primary RBF Kernel')

        [N,~]=size(xTrain);
        [Ntest,~]=size(xTest);
        K=zeros(Ntest,N);
        for i=1:Ntest
            for      j=1:N

                K(i,j)=K(i,j)+...
                    kernelfunction(kerTypeMCM,xTest(i,:),xTrain(j,:),testKerPara);

            end
        end

%         [coeff,score] = pca(K);
% %         subplot(3,2,4);
%         scatter(score(:,1), score(:,2), dotsize, yTest,'filled');
% %         title('Test Data embedding with primary RBF Kernel')



        [ test_pred,testAcc ] = mcmPredict( xTrain,xTest,yTest,kerTypeMCM,testKerPara,lambda,b );
        [ train_pred,trainAcc ] = mcmPredict( xTrain,xTrain,yTrain,kerTypeMCM,testKerPara,lambda,b );

        nsv = length(nonzeros(lambda(lambda>1e-6)));

        bestAccConf=0;
        gam0 = testKerPara;

        for gam = gam0*[2,2.2,2.4,2.6,2.8,3,4,10,15,20,25,30,40,50,60,70,80,100,500,800,1000,5000]
            try
                gam/gam0;
                [testAccConf,trainAccConf] = accMcmConformal(xTrain,yTrain,xTest,yTest,lambda,kerTypeMCM,gam0,gam,Ctest);
                testAccConf; 
            catch
                testAccConf = 0;
                trainAccConf = 0;
            end

            if(testAccConf >= bestAccConf)
                bestAccConf = testAccConf;
                besttrainAccConf = trainAccConf;
                CbestConf = Ctest ;
                bestgam0 = gam0 ;
                bestgam = gam ;
            end
        end 

        [Kt,rtestK,trA,tesA] = plotMcmConformal(xTrain,yTrain,xTest,yTest,lambda,kerTypeMCM,bestgam0,bestgam,CbestConf);
        bestgam0
        bestgam
        CbestConf
        % [coeff,score] = pca(k1);
        % subplot(2,2,3);
        % scatter(score(:,1), score(:,2), dotsize, yTrain,'filled');
%         [coeff,score] = pca(Kt);
% %         subplot(3,2,5);
%         scatter(score(:,1), score(:,2), dotsize, yTrain,'filled');
% %         title('Train Data embedding using proposed kernel optimization')
% 
%         [coeff,score] = pca(rtestK);
% %         subplot(3,2,6);
%         scatter(score(:,1), score(:,2), dotsize, -yTest,'filled');
% %         title('Test Data embedding using proposed kernel optimization')


%         saveas(graphs(fold),sprintf('Results/%d_%d.fig',dataset,fold))
    
    end

end
