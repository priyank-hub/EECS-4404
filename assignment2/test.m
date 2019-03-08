clc;clear
 x = load("bg.txt");
 n = size(x,1);
t_1 = ones(n/2,1);
t_2 = - ones(n/2,1);
t = [t_1;t_2];

SVMModel = fitcsvm(x,t);
L = resubLoss(SVMModel)
L = resubLoss(SVMModel,'LossFun','hinge')
% emp_loss = rand(10,1)
% plot(emp_loss)
% a=rand(5,5)
% a(1,1)
% 
% w = [1 2]
% norm(w)^2
% w * w'