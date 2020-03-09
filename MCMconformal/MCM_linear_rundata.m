clc;
clear all;

result=[];

for dataset=10%change the number from 1 to 30 you can put this in a loop like this
    % for dataset=1:30
    %%%%%%%% call everything here...but I do not recommend it, since debugging
    %%%%%%%% is difficult
    % end
    filename= sprintf('%d.mat',dataset);
    folds=sprintf('%dfold.mat',dataset);
    load(strcat('data_folds/',filename));
    load(strcat('data_folds/',folds));
    clc;
    disp(dataset);
    
    X=x;
    Y=y;
    nfolds=5;
    s=size(X,1);%size of training data
    %hyperparameters to be initialized here
    %degree for polynomial kernel

%     gamma=2.^[-9,-5,-3,-1,1];
    gamma=2.^[-9];
    
%     cParams = 2.^(-25:2:2);
    cParams = 2.^[-9];

    kerTypeMCM ='rbf';

    %also for polynomial kernel
    % beta=1;
    %similarly you can define beta here for RBF kernel and C for cost
    
    t1=[];
    t2=[];
    t3=[];
    
    Cbest=0;
    bestKerPara=0;
    bestAcc=0;
    tic;
    for i=1
%     for i=1:nfolds
        xTrain=[];
        yTrain=[];
        xTest=[];
        yTest=[];

        test = (indices == i);
        train = ~test;
        for j=1:s
            if(train(j)==1)
                xTrain=[xTrain;X(j,:)];
                yTrain=[yTrain;Y(j,:)];
            end
            if(test(j)==1)
                xTest=[xTest;X(j,:)];
                yTest=[yTest;Y(j,:)];
            end
        end
        %data preprocessing
        me=mean(xTrain);
        std_dev=std(xTrain);

        for n=1:size(xTrain,2)
            if(std_dev(n)~=0)
                xTrain(:,n)=(xTrain(:,n)-me(n))./std_dev(n);
            else
                xTrain(:,n)=(xTrain(:,n)-me(n));
            end
        end
        for n=1:size(xTest,2)
            if(std_dev(n)~=0)
                xTest(:,n)=(xTest(:,n)-me(n))./std_dev(n);
            else
                xTest(:,n)=(xTest(:,n)-me(n));
            end
        end
        
        
        [Ctest,testKerPara] = tuneMCM( xTrain, yTrain , kerTypeMCM , cParams , gamma);
        try
            [ lambda,b,h ] = mcm_linear_efs( xTrain, yTrain, kerTypeMCM, testKerPara, Ctest );
        catch
            lambda = rand(size(xTrain,1),1);
            b = rand;
        end
        
        [ test_pred,testAcc ] = mcmPredict( xTrain,xTest,yTest,kerTypeMCM,testKerPara,lambda,b );
        [ train_pred,trainAcc ] = mcmPredict( xTrain,xTrain,yTrain,kerTypeMCM,testKerPara,lambda,b );
      
        nsv = length(nonzeros(lambda(lambda>1e-4)));
        
% conformal
        edindex = lambda>1e-4;
        ed = [xTrain(edindex,:) yTrain(edindex)] ; %empirical data
        data = [xTrain(~edindex,:) yTrain(~edindex)];
        
        [m n] = size(data);
        data = sortrows(data,n);      %sort rows to separate the classes. 
        a = ed(:,1:n-1);
        m2 = sum(data(:,n)+1)/2; % number of points in class m2 = -1 
        m1 = m - m2; 
        de = size(ed,1);
        
        y = data(:,n); 
        p = data(:,1:n-1); 
        
        gam0 = gamma; % set gam0 
        kernel = kerTypeMCM;
        for i = 1:m 
            for j = 1:m 
                K0(i,j) = kernelfunction(kernel, p(i,:), p(j,:),gam0);         % basic kernel 
                if i==j 
                    firstW0(i,j) = K0(i,j);
                else 
                    firstW0(i,j) = 0; 
                end 
            end 
        end 
        
        K11 = K0(1:m1,1:m1);
        K12 = K0(1:m1,m1+1:m);
        K21 = K0(m1+1:m,1:m1);
        K22 = K0(m1+1:m,m1+1:m); 

        testx = xTest;
        testy = yTest;
        
        [m n] = size(p);
        
        gam = gamma*2; % set this larger than gam0
        for i = 1:m 
            for j = 1:de 
                k1matrix(i,j) = kernelfunction(kernel, p(i,:),a(j,:),gam);          %k1 gam = .5
            end 
        end 
        
        e = ones(m,1);
        K1 = [e k1matrix]; 
        B0 = [(1/m1)*K11  zeros(m1,m2);zeros(m2,m1) (1/m2)*K22] - [ (1/m)*K11 (1/m)*K12 ; (1/m)*K21 (1/m)*K22]; 
        W0 = [firstW0]  - [(1/m1)*K11 zeros(m1,m2); zeros(m2,m1) (1/m2)*K22];
        M0 = K1'*B0*K1; 
        N0 = K1'*W0*K1; 
        
        e = eye(m,1);
        C =1e-6; D =1e-6 ;  
        [ralpha lam]  =  eig(K1'*B0*K1+ C*speye(de+1), K1'*W0*K1+D*speye(de+1)) ;
%          check
        max =0 ; maxid =0; 
        for i  = 1: de %changed 
            if(lam(i,i) > max ) 
                max = lam(i,i); 
                maxid = i; 
            end 
        %     fprintf(1,'lam(%d,%d) = %f\n' , i,i,lam(i,i));    
        end

        rJ1 = ralpha(:,maxid)' * (K1'*B0*K1 + C*speye(de+1)) * ralpha(:,maxid) ; 
        rJ2 = ralpha(:,maxid)' * (K1'*W0 *K1+ D*speye(de+1)) * ralpha(:,maxid) ;   

        rJ = rJ1 /rJ2 ;
        qt = K1 * ralpha(:,maxid);

        Kt = zeros(m);
        for  i = 1:m      
            for j = 1:m       
                Kt(i,j) = qt(i) * qt(j) * K0(i,j); 
            end 
        end 
        
        [ lambdaConf,bConf,hConf ] = mcm_linear_efs_conformal( p, y, kerTypeMCM, testKerPara, Ctest, qt );
        [~,trainAccConf] = mcmPredictConformal(p,y,p,y,Kt,lambdaConf,bConf);
        trainAcc
        trainAccConf
        m = size(testx,1);
        qtestr = []; rtestK= []; 
        for  i = 1:m 
            qtestr(i) = ralpha(1,maxid); 
            for j = 1:de 
                qtestr(i)  = qtestr(i) + ralpha(j+1,maxid) * kernelfunction(kernel, testx(i,:), a(j,:), gam); 
            end 
        end 
        for i = 1:m    
            for j = 1: size(p,1) 
                rtestK(i,j) = qtestr(i) * qt(j) * kernelfunction(kernel, testx(i,:), p(j,:), gam0); 
            end     
        end 
        
        [~,testAccConf] = mcmPredictConformal(p,y,testx,testy,rtestK,lambdaConf,bConf);
%         till here
        testAcc
        testAccConf
        
        if(testAcc >= bestAcc)
                bestAcc = testAcc;
                Cbest=Ctest;
                bestKerPara=testKerPara;
        end 
% setting exit as 1 for testing    
        exitFlag=1;
        
        if(exitFlag==1)
            t1=[t1;trainAcc];
            t2=[t2;testAcc];
            t3=[t3;nsv];
        end

    end
    avg1=mean(t1);
    std3=std(t3);
    
    avg2=mean(t2);
    avg3=mean(t3);
    std1=std(t1);
    std2=std(t2);
%   r=[avg1 avg2 avg3 std1 std2 std3 C1(a1) C2(a2) d_min(a3)];
    timeFold = toc;
    
    r=[dataset avg1 std1 avg2 std2 avg3 std3 Cbest bestKerPara timeFold];
    result=[result;r];
    bestAcc
    best_acc = avg2;
    
    fprintf(2,'Best Accuracy :  %.3f     C: %.3f    P: %.3f',best_acc,Cbest,bestKerPara);
%     xlswrite(strcat(int2str(dataset),'result_baseline_gs.xlsx'),result)
end