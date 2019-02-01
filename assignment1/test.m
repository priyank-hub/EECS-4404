clc;clear;close all;

[x_b,t_b] = load_data(["dataset2_inputs.txt","dataset2_outputs.txt"]);
[training, testing] = cross(x_b, t_b, 10);
concat = horzcat(x_b,t_b);
loss_rlm = zeros(25,1);
w_b = zeros(25,21);
for i = 1:25
    w_b(i,:) = rlm_w(training(:,1), training(:,2), 20, -i);
    loss_rlm(i) = q_loss(w_b(i,:), testing(:,1), testing(:,2));
end
% interval = -1:0.01:1.1;
% hold on;
% plot(interval,func(rlm_w(x_b, t_b, 20, -15),interval));
% plot(x_b,t_b,'rx');
% w
plot(loss_rlm);
title('RLM');
ylabel('empirical square loss l');
xlabel('degree W');