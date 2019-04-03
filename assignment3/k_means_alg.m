function [C, cost, init_c] = k_means_alg(D,k,init,init_centers)
% N: the number of points
% d: the dimension of each point
[N, d] = size(D);

% C : indecate each points belong to which class (k=1,2,..or K)
C = zeros(N,1);

% three ways to init:
% 1. simply set the inital centers by hand
% 2. choose k datapoints uniformly at random from the dataset
% 3. choose the first center uniformly at random, and then choose 
%    each next center to be the datapoint that maximizes the sum of
%    (euclidean) distances from the previous datapoints
centers = zeros(k, d); % init k-centers
if (strcmpi(init, 'manual'))
    
    % generate the random number in [p_min,p_max] area
    centers = init_centers;
elseif (strcmpi(init, 'uniform'))
    for i = 1:k
        j = unidrnd(N); % choose index j uniformly at random
        centers(i,:) = D(j,:); % choose jth points uniformly at random
    end  
elseif (strcmpi(init, 'euclidean'))
   % 1st center
   j = unidrnd(N); % choose index j uniformly at random
   centers(1,:) = D(j,:); % choose jth points uniformly at random as first center   
   % 2nd center
   if k >= 2
       previous_point = centers(2,:); 
       % cpmpute the eucliden distances
       distances = zeros(N,1);      
       for n_p = 1:N
           distances(n_p,:) = norm(previous_point-D(n_p));
       end
       % find the max distance point
       max_i = find(distances==max(distances),1);   
       % set this point as point
       centers(2,:) = D(max_i,:);
   end
   % next centers
   if k>= 3
       for i = 3:k
           previous_centers = centers(1:i-1,:);
           % cpmpute the sum of eucliden distances
           distances = zeros(N,1);
           [n_c,~]=size(previous_centers);
           for c = 1:n_c
               for n_p = 1:N
                   distances(n_p,:) = distances(n_p,:) + norm(previous_centers(c,:)-D(n_p,:));
               end
           end
           
           % find the max distance point
           max_i = find(distances==max(distances),1);
           
           % set this point as point
           centers(i,:) = D(max_i,:); 
       end
   end
end
% store inital centers
init_c = zeros(k, d);
init_c = centers;
% repeat until convergence
itr = 0;
max_itr = 100;
while 1
    
   % compute nearest point index
   for i = 1:N
       
       dists = zeros(k,1);
       for j = 1:k
           dists(j,:) = norm(D(i,:)-centers(j,:));
       end
       
       % mark this point which class belongs to
       C(i,1) = find(dists==min(dists),1); 
   end
   
   % store old centers
   old_centers = centers;
   
   % update the center
   for i = 1:k
       neares_i = find(C(:,1)==i);
       centers(i,:) = mean(D(neares_i,:));
   end
    
   % stop when centers are not uptated
   if isequal(old_centers,centers)
       break;
   end
   
   % prevent from dead loop
   itr = itr + 1;
   if itr > max_itr
       break
   end 
end

% cost
% a_cost = 0;
% for i = 1:k
%     cls_i = find(C==i);
%     for j = cls_i'
%         a_cost = a_cost + norm(D(j,:)-centers(i,:));
%     end
% end
b_cost = 0;
for i = 1:N
    b_cost = b_cost + norm(D(i,:) - centers(C(i,:),:))^2;
end
cost = b_cost/N;
    