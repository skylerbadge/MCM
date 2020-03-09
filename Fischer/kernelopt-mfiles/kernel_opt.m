% working SVM version for bc - all three methods 

clear; 
gam = 0.01;
gam0 = 0.0005;
rndloop  =50;

% x = [-5:.1:5]; 
% y = x.^2; 

load 'C:\Users\Skyler\OneDrive\IIT_Delhi\Jayadeva\MCM\Fischer\kerneloptdata\myion.data' \d 
ion1 = myion; 
ion2 = [ion1(:,1), ion1(:,3:end)];      % removing the 2nd const col
data00 = ion2(1:end,1:end); 

% load 'E:\Sameena\SEM2\mlearn repository\breast-cancer-wisconsin\mybcw.data' \d 
% data00 = mybcw(1:end,2:end); 

% load 'E:\Sameena\SEM2\mlearn repository\breast-cancer-30\mywdbc.data' \d 
% data00 = mywdbc(1:end,3:end); 
% % wdbc - 2nd col is the output var. 
% data00 = [data00 mywdbc(:,2)]; 
% gtr = []; gte =[]; str = []; ste=[]; rtr=[]; rte=[]; 
% agtr = 0; agte = 0; astr = 0; aste = 0; artr = 0; arte = 0; 
% 
% [m n] = size(data00);         %m = no of pts, n = no of variables 
% for i = 1:m 
%     if(data00(i,n) == 2) 
%         data00(i,n) = 0; 
%     elseif (data00(i,n) ==4)
%         data00(i,n) = 1; 
%     end
% end

for loop = 1:rndloop
    loop 
eta0 = .01; iter = 200 ;% 200 in paper; 
kernel = 'myrbf' ;
rndstate = loop; 
rand('state',rndstate);

[m n] = size(data00);         %m = no of pts, n = no of variables
mean1 = mean(data00);         
var1 = std(data00); 
for i = 1:m 
    for j = 1:n-1             % last is the class  index 
        data01(i,j) = data00(i,j) - mean1(1,j); 
    end 
end 

for i = 1:m
    for j = 1:n-1
            data02(i,j) = data01(i,j)/var1(1,j);
    end 
end 
data0 = [ data02  data00(:,n)]   ;  % appending the first col ( y ) 

perm = randperm(m); 
newsize = m/3; 
newsize = ceil(newsize);
data = data0(perm(1:newsize),:);
testdata = data0(perm(newsize+1:2*newsize),:);
empericaldata = data0(perm(2*newsize+1:m),:);

[m n] = size(data);
data = sortrows(data,n);      %sort rows to separate the classes. 
a = empericaldata(:,1:n-1);
m2 = sum(data(:,n)); 
m1 = m - m2; 
de = size(empericaldata,1);
    
y = data(:,n); 
p = data(:,1:n-1); 
for i = 1:m
    if y(i) == 0 
        y(i) = -1; 
    end 
end 

for i = 1:m 
    for j = 1:m 
        K0(i,j) = feval(kernel, p(i,:), p(j,:),gam0);         % basic kernel 
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

testx = testdata(:,1:end-1); 
testy = testdata(:,end);
for i = 1:m 
    if testy(i) == 0 
        testy(i) = -1; 
    end 
end 

[m n] = size(p);
for i = 1:m 
    for j = 1:de 
        k1matrix(i,j) = feval(kernel, p(i,:),a(j,:),gam);          %k1 gam = .5
    end 
end 

e = ones(m,1);
K1 = [e k1matrix]; 
B0 = [(1/m1)*K11  zeros(m1,m2);zeros(m2,m1) (1/m2)*K22] - [ (1/m)*K11 (1/m)*K12 ; (1/m)*K21 (1/m)*K22]; 
W0 = [firstW0]  - [(1/m1)*K11 zeros(m1,m2); zeros(m2,m1) (1/m2)*K22];
M0 = K1'*B0*K1; 
N0 = K1'*W0*K1; 

alpha = [1 zeros(de,1)']';
q = zeros (m,1); 

for t = 1 : iter              % taking variable t instead of n used in paper.
    q = zeros (m,1); 
    q = K1 * alpha;   
    J1 = q' * B0 * q ; 
    J2 = q' * W0 * q ;   
    J = J1 /J2 ;     
	eta = eta0*(1 - (t-1)/iter);
    alpha=alpha+(eta/J2/J2)*(J2*M0-J1*N0)*alpha;
    alpha=(1.0/sqrt(alpha'*alpha))*alpha;   
end 
q = K1 * alpha;
K = zeros(m);
for  i = 1:m      
    for j = 1:m       
        K(i,j) = q(i) * q(j) * K0(i,j); 
    end 
end 

% using steve gunn's 
global p1 
p1 = gam0; 

[nsv, w1, gam1] = svc1(p,y,'rb',1000); 
nerrs = svcerror(p,y,p,y,K0,w1,gam1) ;
gtr(loop) = nerrs/m * 100; 
fprintf(1,'k0 kernel - training error is %f\n', gtr(loop) ); 

testK0 = []; 
for i = 1: m 
    for j = 1:m 
        testK0(i,j) = feval(kernel, testx(i,:), p(j,:), gam0); 
    end 
end 
nerrs = svcerror(p,y,testx,testy,testK0,w1,gam1) ;
gte(loop) = nerrs/m * 100; 
fprintf(1,'k0 kernel - testing error is %f\n', gte(loop) ); 
% ---- % 

[nsv, w2, gam2] = svc1(p,y,'rb',1000, q); 
nerrs = svcerror(p,y,p,y,K,w2,gam2) ;
str(loop) = nerrs/m * 100; 
fprintf(1,'swamy - training error is %f\n', str(loop) ); 


qtest = []; 
for  i = 1:m 
    qtest(i) = alpha(1); 
    for j = 1:de 
        qtest(i)  = qtest(i) + alpha(j+1) * feval(kernel, testx(i,:), a(j,:), gam); 
    end 
end  % to be used for passing in coresponding svcerror
    
for i = 1:m
    for j = 1:m 
        stestK(i,j) = qtest(i) * q(j) * feval(kernel, testx(i,:), p(j,:), gam0); 
    end     
end 

nerrs = svcerror(p,y,testx,testy,stestK,w2,gam2) ;
ste(loop) = nerrs/m * 100; 
fprintf(1,'swamy - testing error is %f\n', ste(loop) ); 

e = eye(m,1);
C =1e-6; D =1e-6 ; 
clear lam; 
clear rq; 

[ralpha lam]  =  eig(K1'*B0*K1+ C*speye(de+1), K1'*W0*K1+D*speye(de+1)) ; 
max =0 ; maxid =0; 
for i  = 1: m 
    if(lam(i,i) > max ) 
        max = lam(i,i); 
        maxid = i; 
    end 
%     fprintf(1,'lam(%d,%d) = %f\n' , i,i,lam(i,i));    
end

rJ1 = ralpha(:,maxid)' * (K1'*B0*K1 + C*speye(de+1)) * ralpha(:,maxid) ; 
rJ2 = ralpha(:,maxid)' * (K1'*W0 *K1+ D*speye(de+1)) * ralpha(:,maxid) ;   
J;
rJ = rJ1 /rJ2 ;
qt = K1 * ralpha(:,maxid);

Kt = zeros(m);
for  i = 1:m      
    for j = 1:m       
        Kt(i,j) = qt(i) * qt(j) * K0(i,j); 
    end 
end 

[nsv, w3, gam3] = svc1(p,y,'rb',1000, qt); 
nerrs = svcerror(p,y,p,y,Kt,w3,gam3) ;
rtr(loop) = nerrs/m * 100; 
fprintf(1,'our - training error is %f\n', rtr(loop) ); 

qtestr = []; rtestk= []; 
for  i = 1:m 
    qtestr(i) = ralpha(1,maxid); 
    for j = 1:de 
        qtestr(i)  = qtestr(i) + ralpha(j+1,maxid) * feval(kernel, testx(i,:), a(j,:), gam); 
    end 
end 
for i = 1:m    
    for j = 1: size(testy,1) 
        rtestK(i,j) = qtestr(i) * qt(j) * feval(kernel, testx(i,:), p(j,:), gam0); 
    end     
end 
nerrs = svcerror(p,y,testx,testy,rtestK,w3,gam3) ;
rte(loop) = nerrs/m * 100; 
fprintf(1,'our - testing error is %f\n', rte(loop) ); 
end 
svmc=1000;
d = datestr(now);    
d1 = datestr(d,30); 
s1 = sprintf('E:\\Sameena\\sem2\\generalized\\newoutputs\\%s.wordpad',d1);
fid = fopen(s1,'w'); 
fprintf(fid, 'database = ionosphere\n ');
fprintf(fid, 'C = %f, D = %f, svmc= %d, eta = %f\nkernel=%s\ngam = %f\ngam0 = %f\n' , C, D, svmc, eta, kernel,gam, gam0);
fprintf(fid,'rnd_seed simple_training_error simple_testing_error swamy_training_erro swamy_testing_error rayleigh_training_error rayleigh_testing_error \n');    
for i = 1:rndloop
    fprintf(fid, ' %d %f  %f  %f  %f  %f  %f\n', i, gtr(i), gte(i), str(i), ste(i), rtr(i), rte(i)); 
end 
    
% gtr
% gte
% str
% ste
% rtr
% rte
fprintf(1,'simple average\n' ); 
agtr = mean(gtr)
agte = mean(gte)
fprintf(1,'swamy average\n' ); 
astr = mean(str)
aste = mean(ste)
fprintf(1,'our average\n' ); 
artr = mean(rtr)
arte = mean(rte)
gam 
gam0
fprintf(fid, '\n\nAverage errors %f  %f  %f  %f  %f  %f\n', agtr, agte, astr, aste, artr, arte); 
fprintf(fid, '\nTotal generalization error = %f   %f  %f\n', agtr+agte, astr+ aste, artr+ arte); 
fclose(fid);
