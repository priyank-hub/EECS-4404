clc;clear;close all;

% [x_b,t_b] = load_data(["dataset2_inputs.txt","dataset2_outputs.txt"]);
x_b = load("dataset2_inputs.txt");
t_b = load("dataset2_outputs.txt");

avg_loss = cross_vailidation(x_b,t_b,5,10);
% d = 5;
% fold = 10;
% 
% %[training, testing] = cross(x_b, t_b, 10);
% % x_b = [1 3 3 4 5 6 7 8 9 10];
% % x_b=x_b';
% % t_b = [11 12 13 14 15 16 17 18 19 20];
% % t_b=t_b';
% concat = horzcat(x_b,t_b);
% rowrank = randperm(size(concat, 1));
% rank_data = concat(rowrank, :);
% 
% chunck = size(rank_data,1)/fold; % the number of times of testing
% training = zeros((fold-1) * chunck, 2); % init size of training
% testing = zeros(chunck, 2); % init size of testing
% loss = 0;
% for i = 1:fold
%     n=1;
%     testing = zeros(chunck, 2);
%     % load testing set
%     for j = 1+(i-1)*chunck:i*chunck
%         testing(n,:) = rank_data(j,:);
%         n=n+1;
%     end
%     % load remaining rank_data for training set
%     training = rank_data(~ismember(rank_data,testing,'rows'),:);
%     
%     % training our model
%     w = erm_w(training(:,1), training(:,2), d);
%     
%     % compute the total loss
%     loss = loss + q_loss(w, testing(:,1), testing(:,2));
% end
% loss = loss/fold;

















% loss_rlm = zeros(25,1);
% w_b = zeros(25,21);
% for i = 1:25
%     w_b(i,:) = rlm_w(training(:,1), training(:,2), 20, -i);
%     loss_rlm(i) = q_loss(w_b(i,:), testing(:,1), testing(:,2));
% end
% % interval = -1:0.01:1.1;
% % hold on;
% % plot(interval,func(rlm_w(x_b, t_b, 20, -15),interval));
% % plot(x_b,t_b,'rx');
% % w
% plot(loss_rlm);
% title('RLM');
% ylabel('empirical square loss l');
% xlabel('degree W');


