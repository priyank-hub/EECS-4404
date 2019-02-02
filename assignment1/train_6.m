function [opt_w,min_loss] = train_6 (x_b,t_b)
% x_b = load("dataset2_inputs.txt");
% t_b = load("dataset2_outputs.txt");
%concat pair of inputs and outputs
concat = horzcat(x_b,t_b);

% rank data randomly
rowrank = randperm(size(concat, 1));
rank_data = concat(rowrank, :);

% init some para.
fold = 10;
chunck = size(rank_data,1)/fold; % the number of times of testing
min_loss = inf;

% compute loss with cross vailidation, which is with each degree W
loss = zeros(1,fold);
for degree = 1:20
    w = zeros(degree+1,fold);
    for i = 1:fold
        n=1;
        testing = zeros(chunck, 2);
        % load testing set
        for j = 1+(i-1)*chunck : i*chunck
            testing(n,:) = rank_data(j,:);
            n=n+1;
        end
        % load remaining rank_data for training set
        training = rank_data(~ismember(rank_data,testing,'rows'),:);
        
        % training our model
        w(:,i) = erm_w(training(:,1), training(:,2), degree);
        
        % compute the total loss
        loss(:,i) = q_loss(w(:,i), testing(:,1), testing(:,2));
        
        n = 1;
        loss_rlm = zeros(1,20);
        w_rlm = zeros(degree+1,20);
        for ln_lambda = 1:20
            
            % training our model
            w_rlm(:,n) = rlm_w(training(:,1), training(:,2), degree, -ln_lambda);
            
            % compute the total loss
            loss_rlm(:,n) = q_loss(w_rlm(:,n), testing(:,1), testing(:,2));
            n=n+1;
        end
        if min(loss_rlm) < min_loss
            min_loss = min(loss_rlm);
            index = loss_rlm==min(loss_rlm);
            opt_w = w_rlm(:,index);
            %flag="RLM";
        end
        
    end
    if min(loss) < min_loss
        min_loss = min(loss);
        index = loss==min(loss);
        opt_w = w(:,index);
        %flag="EMR";
    end
end
