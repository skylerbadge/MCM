function [f,g] = mysim(z,model);

    n = model.x_dim; 
    npts = model.nb_data; 
    Xt = z;     
    Yt = [] ; 
    
    
    
    if size(model.alpha,1) ~= npts 
        disp('error:');
    end 
    
    
    if model.preprocess(1)=='p',
        [Xt, Yt] = prelssvm(model,Xt,Yt);
    end
    sig2 = model.kernel_pars;
    if size(sig2)<2 
        t = 1;
        degree =1 ;
    else 
        t = sig2(1); 
        degree = sig2(2); 
    end 
    

    temp =model.b ;   
    for i=1:npts
        temp2 = ( t+ model.xtrain(i,:)*(Xt'));
        temp = temp + model.alpha(i,:)*temp2^degree;
    end 
    f = temp; 

    temp1 = zeros(1,n);
    if model.preprocess(1)=='p' & ~(model.type(1)=='c' & strcmp(model.latent,'yes')),
        [X,f] = postlssvm(model,[],f);
    end

    for i = 1:npts
        temp2 = ( t+ model.xtrain(i,:)*(Xt'));
        temp2 = temp2^(degree-1);         
        temp1(i,:) =  degree*model.alpha(i,:)*temp2*model.xtrain(i,:); 
    end
    g = sum(temp1);   

%    
 %   if model.preprocess(1)=='p' & ~(model.type(1)=='c' & strcmp(model.latent,'yes')),
  %      for i =1:n 
   %         [X,g(1,i)] = postlssvm(model,[],g(1,i));
   %    end 
   % end


    
    
%    if model.preprocess(1)=='p' & ~(model.type(1)=='c' & strcmp(model.latent,'yes')),
 %       [X,g] = postlssvm(model,[],g);
 %  end
