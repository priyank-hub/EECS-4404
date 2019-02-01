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

% STEP 4 cross vailidation. Implement 10-fold cross validation for ERM.
[training, testing] = cross(x, t, 10);
loss_cross_val = zeros(20,1);
for d = 1:20
    w = erm_w(training(:,1), training(:,2), d);
    %loss_cross_val(d) = q_loss(w, training(:,1), training(:,2));
    loss_cross_val(d) = q_loss(w, testing(:,1), testing(:,2));
end
plot(loss_cross_val);
title('cross vailidation for ERM');
ylabel('empirical square loss l');
xlabel('degree W');
fprintf("step4: Finish cross vailidation and plot empirical square loss on the data, please press enter to continue!\n")
pause;
clc;close all;

% STEP 5 visualization. 
% init some setting
degrees = [1 5 10 20];
interval = -1.05:0.01:1.05;
label = string(zeros(2*length(degrees)+1,1));

% plot the data along with the ERM learned models
hold on
for d = degrees
    w_erm_vis = erm_w(x, t, d);
    plot(interval,func(w_erm_vis,interval));
end
% plot the data along with the RLM learned models
for d = [1 5 10 20]
    w_rlm_vis = rlm_w(x, t, 20, log(0.001));
    plot(interval,func(w_rlm_vis,interval));
end
% plot origin dataset
plot(x,t,'rx');

% load labels
n = 1;
for i = degrees
    label(n) = ("W = " +num2str(i)+" (ERM)");
    n = n + 1;
end
for i = degrees
    label(n) = ("W = " +num2str(i)+" (RLM)");
    n = n + 1;
end
label(n) = "datapoints";
title('Visualization ERM vs RLM vs OriginDataPoints');
ylabel('outputs t');
xlabel('inputs x');
legend(label');
fprintf("step5: Finish visualization with ERM and RLM, please press enter to continue!\n")
pause;
clc;close all;






