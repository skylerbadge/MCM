function [testAccConf, trainAccConf] = accMcmConformal(xTrain,yTrain,xTest,yTest,lambda,kerTypeMCM,gam0,gam,Cpara)
    
        edindex = abs(lambda)>1e-6;
%         edindex =  edindex0 & (trainPred==yTrain);  
        ed = [xTrain(edindex,:) yTrain(edindex)] ; %empirical data
%         data = [xTrain(~edindex,:) yTrain(~edindex)];
        data = [xTrain yTrain];
        
        [m n] = size(data);
        data = sortrows(data,n);      %sort rows to separate the classes. 
        a = ed(:,1:n-1);
        m2 = sum(data(:,n)+1)/2; % number of points in class m2 = -1 
        m1 = m - m2; 
        de = size(ed,1);
        
        y = data(:,n); 
        p = data(:,1:n-1); 
        
        kernel = kerTypeMCM;
        K0 = zeros(m,m);
        firstW0 = zeros(m,m);
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
        
        [m n] = size(p);

        k1matrix = zeros(m,de);
        for i = 1:m 
            for j = 1:de 
                k1matrix(i,j) = kernelfunction(kernel, p(i,:),a(j,:),gam);          %k1 gam = .5
            end 
        end 
        
        e = ones(m,1);
        K1 = [e k1matrix]; 
        B0 = [(1/m1)*K11  zeros(m1,m2);zeros(m2,m1) (1/m2)*K22] - [ (1/m)*K11 (1/m)*K12 ; (1/m)*K21 (1/m)*K22]; 
        W0 = [firstW0]  - [(1/m1)*K11 zeros(m1,m2); zeros(m2,m1) (1/m2)*K22];

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

%         rJ1 = ralpha(:,maxid)' * (K1'*B0*K1 + C*speye(de+1)) * ralpha(:,maxid) ; 
%         rJ2 = ralpha(:,maxid)' * (K1'*W0 *K1+ D*speye(de+1)) * ralpha(:,maxid) ;   
% 
%         rJ = rJ1 /rJ2 ;
        qt = K1 * ralpha(:,maxid);

        Kt = zeros(m);
        for  i = 1:m      
            for j = 1:m       
                Kt(i,j) = qt(i) * qt(j) * K0(i,j); 
            end 
        end 
        
        [ lambdaConf,bConf,hConf ] = mcm_linear_efs_conformal( p, y, kerTypeMCM, gam0, Cpara, qt ); % check gam
        [~,trainAccConf] = mcmPredictConformal(p,y,p,y,Kt,lambdaConf,bConf);
        m = size(xTest,1);
        
        qtestr = zeros(1,m);
        rtestK= zeros(m,size(p,1)); 
        for  i = 1:m 
            qtestr(i) = ralpha(1,maxid);  
            for j = 1:de 
                qtestr(i)  = qtestr(i) + ralpha(j+1,maxid) * kernelfunction(kernel, xTest(i,:), a(j,:), gam); 
            end 
        end 
        for i = 1:m    
            for j = 1: size(p,1) 
                rtestK(i,j) = qtestr(i) * qt(j) * kernelfunction(kernel, xTest(i,:), p(j,:), gam0); 
            end     
        end 
        [~,testAccConf] = mcmPredictConformal(p,y,xTest,yTest,rtestK,lambdaConf,bConf);
end