function avg_loss = cross_vailidation_rlm(rank_data,degree,lambdaln_lambda,fold)

% % concat pair of inputs and outputs
% concat = horzcat(x,t);
% 
% % rank data randomly
% rowrank = randperm(size(concat, 1));
% rank_data = concat(rowrank, :);

chunck = size(rank_data,1)/fold; % the number of times of testing
tot_loss = 0; % init total loss
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
    w = rlm_w(training(:,1), training(:,2), degree, lambdaln_lambda);
    
    % compute the total loss
    tot_loss = tot_loss + q_loss(w, testing(:,1), testing(:,2));
end
avg_loss = tot_loss/fold;