clc;clear;close all;

% STEP 1 load data
[x, t] = load_data(["dataset2_inputs.txt","dataset2_outputs.txt"]);
fprintf("step1: Finish load data and plot, please press enter to continue!\n")
pause
clc;close all;

% STEP 2 minimizer of the empirical risk (ERM). compute the empirical square 
% loss (Loss) on the data and plot it as a function of degree W
loss_erm = zeros(20,1);

% compute weight with ERM and loss, which is with each degree W
for d = 1:20
    w = erm_w(x, t, d);
    loss_erm(d) = q_loss(w, x, t);
end

% Normalization for loss
loss_erm = loss_erm/max(loss_erm);

% plot the loss_erm graph with degree W
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
% compute weight with RLM and loss, which is with each degree W
for i = 1:20
    w = rlm_w(x, t, 20, -i);
    loss_rlm(i) = q_loss(w, x, t);
end

% Normalization for loss
loss_rlm = loss_rlm/max(loss_rlm);

% plot the loss_rlm graph with degree W
plot(loss_rlm);
title('RLM');
ylabel('empirical square loss l');
xlabel('i (indicate:ln(lambda))');
fprintf("step3.1: Finish compute and plot empirical square loss on the data, please press enter to continue!\n")
pause;
plot(loss_erm);
hold on
plot(loss_rlm);
legend('EMR', 'RLM');
title('ERM v.s RLM');
ylabel('empirical square loss l');
xlabel('W/i');
fprintf("step3.2: Finish compare ERM and RLM, please press enter to continue!\n")
pause;
clc; close all;

% STEP 4 cross vailidation. Implement 10-fold cross validation for ERM.

% concat pair of inputs and outputs
concat = horzcat(x,t);

% % rank data randomly
% rowrank = randperm(size(concat, 1));
% rank_data = concat(rowrank, :);

% init some para.
loss_cross_val = zeros(20,1);
fold = 10;

% compute loss with cross vailidation, which is with each degree W
for d = 1:14
    % rank data randomly
    rowrank = randperm(size(concat, 1));
    rank_data = concat(rowrank, :);

    loss_cross_val(d) = cross_vailidation_erm(rank_data,d,fold);
end

% Normalization for loss
loss_cross_val = loss_cross_val/max(loss_cross_val);

% plot the loss_cross_val graph with degree W
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
interval = -1:0.01:1.05;
label_erm = string(zeros(length(degrees)+1,1));
label_rlm = string(zeros(length(degrees)+1,1));

% load labels
n = 1;
for i = degrees
    label_erm(n) = ("W = " +num2str(i)+" (ERM)");
    n = n + 1;
end
label_erm(n) = "datapoints";
n = 1;
for i = degrees
    label_rlm(n) = ("W = " +num2str(i)+" (RLM)");
    n = n + 1;
end
label_rlm(n) = "datapoints";

% plot the data along with the ERM learned models
figure;
subplot(1,2,1)
hold on;
for d = degrees
    w_erm_vis = erm_w(x, t, d);
    plot(interval,func(w_erm_vis,interval));
end
plot(x,t,'rx');

title('Visualization ERM vs OriginDataPoints');
ylabel('outputs t');
xlabel('inputs x');
legend(label_erm');

subplot(1,2,2)
hold on
% plot the data along with the RLM learned models
for d = [1 5 10 20]
    w_rlm_vis = rlm_w(x, t, d, log(0.001));
    plot(interval,func(w_rlm_vis,interval));
end
% plot origin dataset
plot(x,t,'rx');

title('Visualization RLM (with lambda=0.001) vs OriginDataPoints');
ylabel('outputs t');
xlabel('inputs x');
legend(label_rlm');

fprintf("step5: Finish visualization with ERM and RLM, please press enter to continue!\n")
pause;
clc;close all;

% % STEP 6 bonus
% x_b = load("dataset2_inputs.txt");
% t_b = load("dataset2_outputs.txt");
% interval = -1:0.01:1.25;
% 
% [opt_w, min_loss] = train_6(x_b,t_b);
% for i = 1:5
%     [w,l] = train_6(x_b,t_b);
%     if l < min_loss 
%         min_loss = l;
%         opt_w = w;
%     end
% end
% 
% opt_w
% 
% plot(interval,func(opt_w,interval));
% hold on
% plot(x_b,t_b,'rx');
% title('Visualization Model vs OriginDataPoints');
% ylabel('outputs t');
% xlabel('inputs x');
% save('opt_w_s6.txt','opt_w');
% fprintf("step6: Finish training model, please press enter to continue!\n")
% pause;
% clc;close all;
% 
