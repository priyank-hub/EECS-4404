clc;clear;close all;

% STEP 1 load data
[x, t] = load_data;
fprintf("step1: Finish load data and plot, please press enter to continue!\n")
pause;

% STEP 2 compute the empirical square loss (Loss) on the data 
% and plot it as a function of degree W
loss = zeros(20,1);
for d = 1:20
    w = erm_w(x, t, d);
    loss(d) = q_loss(w, x, t);
end
plot(loss);
title('ERM');
ylabel('empirical square loss l');
xlabel('degree W');
fprintf("step2: Finish compute and plot empirical square loss on the data, please press enter to continue!\n")
pause;

% STEP 3