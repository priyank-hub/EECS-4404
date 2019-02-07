% STEP 4 cross vailidation. Implement 10-fold cross validation for ERM.
x = load("dataset1_inputs.txt");
t = load("dataset1_outputs.txt");
% concat pair of inputs and outputs
concat = horzcat(x,t);

% rank data randomly
rowrank = randperm(size(concat, 1));
rank_data = concat(rowrank, :);

% init some para.
loss_cross_val = zeros(20,1);
fold = 10;

% compute loss with cross vailidation, which is with each degree W
for d = 1:20
    % rank data randomly
%     rowrank = randperm(size(concat, 1));
%     rank_data = concat(rowrank, :);

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