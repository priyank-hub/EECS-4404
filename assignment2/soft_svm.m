function [w, hi_loss, bi_loss] = soft_svm(D,T,lambda, n)
% Initialize
[N, clo] = size(D);
x = D(:,1:clo-1); % features
t = D(:,clo); % label {-,+}
theta=zeros(1,clo-1);

hi_loss = zeros(T,1);
bi_loss = zeros(T,1);

% store `w` in each iteration
ws=zeros(T,clo-1);

w_j = (1/(lambda))*theta;

for j = 1:T
    
%     % update `w`
%     w_j = (1/(j*lambda))*theta;
%     
%     % store the `w` in `ws`
%     ws(j,:) = w_j;
 
    % Choose the i in uniform distribution
    i = unidrnd(N);
    
    if t(i) * dot(w_j, x(i,:)) < 1
        
        theta = theta + t(i)*(x(i,:));
        
    end
    
    % update `w`
    w_j = (1/(j*lambda))*theta;
    
    % store the `w` in `ws`
     ws(j,:) = w_j;
    
    % Track the emprical and hinge loss
    hi_loss(j,:) = emp_loss(w_j, D, 'hinge');
    bi_loss(j,:) = emp_loss(w_j, D, 'binary');
end

if(n ~= 0)
    w = (1/n) * sum(ws((T-n:T),:));
else
    w = w_j;
end





k=find(bi_loss==min(bi_loss));
w=ws(k(1),:);
