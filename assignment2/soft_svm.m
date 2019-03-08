function [result,emp_loss,hi_loss] = soft_svm(D,T,lambda)
% clc;clear
% The number of instance in the dataset D
%  D = load_data("bg.txt");
%  T=500
%  lambda = 10

% Initialize
[N, clo] = size(D);
x = D(:,1:clo-1);
t = D(:,clo);
theta=zeros(1,clo-1);
emp_loss = zeros(T+1,1);
hi_loss = zeros(T+1,1);

for j = 1:T
    
    w = (1/(j*lambda))*theta
 
    % Choose the i in uniform distribution
    i = unidrnd(N);
    
    if t(i) * dot(w, x(i,:)) < 1
        
        theta = theta + t(i)*(x(i,:));
        
    end
    
    % Track the emprical and hinge loss
    emp_loss(j,:) = emprical_hinge_loss(w, x, t, lambda);
    hi_loss(j,:) = hinge_loss(w, x(i,:), t(i));
end
result = w;
