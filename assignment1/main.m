clc;clear;close all;

% STEP 1 load data
[x, t] = load_data;
fprintf("step1: Finish load data and plot, please press enter to continue!\n")
pause
clc;close all;

% STEP 2 minimizer of the empirical risk (ERM). compute the empirical square 
% loss (Loss) on the data and plot it as a function of degree W
loss_erm = zeros(20,1);
for d = 1:20
    w = erm_w(x, t, d);
    loss_erm(d) = q_loss(w, x, t);
end
plot(loss_erm);
title('ERM');
ylabel('empirical square loss l');
xlabel('degree W');
fprintf("step2: Finish compute and plot empirical square loss on the data, please press enter to continue!\n")
pause;
clc;close all;

% STEP 3 minimizer of the regularized risk (RLM). compute regularized least
% squares regression on the data and plot the empirical loss as a function
% of i. compare ERM and RLM
loss_rlm = zeros(20,1);
for i = 1:20
    w = rlm_w(x, t, 20, -i);
    loss_rlm(i) = q_loss(w, x, t);
end
plot(loss_rlm);
title('RLM');
ylabel('empirical square loss l');
xlabel('degree W');
fprintf("step3.1: Finish compute and plot empirical square loss on the data, please press enter to continue!\n")
pause;
plot(loss_erm);
hold on
plot(loss_rlm);
legend('EMR', 'RLM');
title('ERM v.s RLM');
ylabel('empirical square loss l');
xlabel('d/i');
fprintf("step3.2: Finish compare ERM and RLM, please press enter to continue!\n")
pause;
clc; close all;

% STEP 4 cross vailidation 
