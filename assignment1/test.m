clc;clear;close all;

% % [x_b,t_b] = load_data(["dataset2_inputs.txt","dataset2_outputs.txt"]);
% x_b = load("dataset2_inputs.txt");
% t_b = load("dataset2_outputs.txt");
% %concat pair of inputs and outputs
% concat = horzcat(x_b,t_b);
% 
% % rank data randomly
% rowrank = randperm(size(concat, 1));
% rank_data = concat(rowrank, :);
% 
% % init some para.
% loss_cross_val = zeros(20,1);
% fold = 10;
% chunck = size(rank_data,1)/fold; % the number of times of testing
% min_loss = inf;
% 
% % compute loss with cross vailidation, which is with each degree W
% loss = zeros(1,fold);
% for degree = 1:20
%     w = zeros(degree+1,fold);
%     for i = 1:fold
%         n=1;
%         testing = zeros(chunck, 2);
%         % load testing set
%         for j = 1+(i-1)*chunck : i*chunck
%             testing(n,:) = rank_data(j,:);
%             n=n+1;
%         end
%         % load remaining rank_data for training set
%         training = rank_data(~ismember(rank_data,testing,'rows'),:);
%         
%         % training our model
%         w(:,i) = erm_w(training(:,1), training(:,2), degree);
%         
%         % compute the total loss
%         loss(:,i) = q_loss(w(:,i), testing(:,1), testing(:,2));
%         
% %         n = 1;
% %         loss_rlm = zeros(1,20);
% %         w_rlm = zeros(degree+1,20);
% %         for ln_lambda = 1:20
% %             
% %             % training our model
% %             w_rlm(:,n) = rlm_w(training(:,1), training(:,2), degree, -ln_lambda);
% %             
% %             % compute the total loss
% %             loss_rlm(:,n) = q_loss(w_rlm(:,n), testing(:,1), testing(:,2));
% %             n=n+1;
% %         end
% %         if min(loss_rlm) < min_loss
% %             min_loss = min(loss_rlm);
% %             index = find(loss_rlm==min(loss_rlm));
% %             opt_w = w_rlm(:,index);
% %             flag="RLM";
% %         end
%         
%     end
%     if min(loss) < min_loss
%         min_loss = min(loss);
%         index = find(loss==min(loss));
%         opt_w = w(:,index);
%         flag="EMR";
%     end
% end

% w_erm=zeros(21,20);
% loss_erm=zeros(1,20);
% w_rlm=zeros(21,20);
% loss_rlm=zeros(1,20);
% 
% for d = 1:20
%     w_erm(1:d+1,d) = erm_w(x_b, t_b, d);
%     w_rlm(:,d) = rlm_w(x_b, t_b, 20, -d);
%     loss_erm(:,d) = q_loss(w_erm(:,d), x_b, t_b);
%     loss_rlm(:,d) = q_loss(w_rlm(:,d), x_b, t_b);
% end
% if min(loss_erm) < min(loss_rlm)
%         min_loss = min(loss_erm);
%         index = find(loss_erm==min(loss_erm));
%         opt_w = w_erm(:,index);
%         flag="EMR";
% else
%         min_loss = min(loss_rlm);
%         index = find(loss_rlm==min(loss_rlm));
%         opt_w = w_rlm(:,index);
%         flag="RLM";
%         
% end
x_b = load("dataset2_inputs.txt");
t_b = load("dataset2_outputs.txt");

[opt_w, min_loss] = train_6(x_b,t_b);
for i = 1:5
    [w,l] = train_6(x_b,t_b);
    if l < min_loss 
        min_loss = l;
        opt_w = w;
    end
end

% flag
degree=size(opt_w,1)
interval = -1:0.01:1.25;
opt_w_a = [0.6006;1.1576;-11.8218;9.7903;33.6301;-27.3195;-22.0138;16.8897];

p = polyfit(x_b,t_b,8);
plot(interval,polyval(p,interval));
hold on
plot(interval,func(opt_w,interval));

plot(interval,func(opt_w_a,interval));
legend("fit","w","w_a");
plot(x_b,t_b,'rx');

% x = [1 2;2 2;3 2;1 2;5 2;3 6;4 7]
% b = [1;0;0;0;0;1;1];
% a = x==min(x(1,:))
% c=x(a,:)