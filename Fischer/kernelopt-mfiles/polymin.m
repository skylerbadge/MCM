function [localmin2,val2,flag] = polymin(x,y,startstate);
    global filecount
    type = 'function estimation';
    gam = 1000;
    degree = 2; 
    t = 1; 
    
    kernel = 'poly_kernel';
%    sig2 = .2;
    sig2 = [t degree];
    tot = 0 ;
    flag =0 ; 
        
    model = initlssvm(x,y,type,gam,sig2,kernel,'preprocess');
    model.implementation = 'matlab';
    model = trainlssvm(model);
    model.alpha 
    model.b
    for i = model.nb_data 
        tot = tot + abs(model.alpha(i)) ; 
    end 
    
    if tot <.001 
        flag =1 ; 
    end 
        
    options = optimset('GradObj','on','maxfunevals',100,'maxiter',100,'tolx',1e-6,'tolfun',1e-6);
    %val = feval('minimise',xt,model);
    s1 = sprintf('C:\\modelfile%d.mat',filecount);
    exit =0 ; 
    n = model.nb_data; 
    stcnt = 0;
    
%    while exit ==0 & stcnt ~=n-1

        av = sum(model.xtrain)/n ;            
%        startpt = model.xtrain(model.nb_data - stcnt); 
        startpt = av; 
        [localmin2, val2,exit,output,grad] = fminunc('mysim',startstate,options,model)
%        stcnt = stcnt + 1; 
        %   end 
        
        
%    if localmin - model.b < localmin/5 
 %       flag = 1 ;
 %  end 
 
    save(s1,'model');
    filecount = filecount + 1 ;
     