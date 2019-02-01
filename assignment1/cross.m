function [training, testing] = cross(x,y,fold)

% rank the data randomly
concat = horzcat(x,y);
rowrank = randperm(size(concat, 1));
rank_data = concat(rowrank, :);

% divide the data into training and testing set
% last data chunck in testing set, remaining in training set
n = 1;m = 1;
chunck = size(rank_data,1)/fold;
training = zeros((fold-1) * chunck, 2);
testing = zeros(size(rank_data,1) - ((fold-1) * chunck), 2);
for i = 1:size(rank_data,1)
    if i <= (fold-1) * chunck
        training(n,:) = rank_data(i,:);
        n = n + 1;
    else
        testing(m,:) = rank_data(i,:);
        m = m + 1;
    end
end
