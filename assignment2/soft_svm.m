function w = soft_svm(D,T,lambda,n)
clc;clear
% The number of instance in the dataset D
%  D = load_data("bg.txt");
%  T=500
%  lambda = 10

% Initialize
[N, clo] = size(D);
x = D(:,1:clo-1);
t = D(:,clo);
theta=zeros(1,clo-1);

for j = 0:T
    
    w = (1/lambda)*theta;
 
    % Choose the i in uniform distribution
    i = unidrnd(N);
    
    if t(i) * dot(w, x(i,:)) < 1
        
        theta = theta + t(i)*(x(i,:));
        
    end
    
    % compute the emp_loss
    emprical_hinge_loss(w, x, t, lambda)
end

