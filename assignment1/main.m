clc;clear;close all;

% STEP 1 load data
[x, t] = load_data;
fprintf("step1: Finish load data and plot, please press enter to continue!\n")
pause;

% STEP 2 compute the empirical square loss (Loss) on the data 
% and plot it as a function of degree w
l = zeros(20,1);
for d = 1:20
    w = ERM(x, t, d);
    x2 = -1.5:0.01:1.5;
    Loss = q_loss_func(w, x, t);
    l(d) = Loss;
    plot(d, Loss, 'r*');
    hold on
end
plot(l);
title('ERM');
ylabel('empirical square loss l');
xlabel('degree W');
fprintf("step2: Finish compute and plot empirical square loss on the data, please press enter to continue!\n")
pause;

% STEP 3