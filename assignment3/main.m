% clc;clear;close all
% % (b)-(c) load 'twodpoints.txt' and plot
% 
% % load data
% twoD=load('twodpoints.txt');
% [N, d] = size(twoD);
% x = twoD(:,1);
% t = twoD(:,2);
% 
% % % plot origin dataset
% % figure(1);
% % %subplot(1,2,1);
% % plot(x, t, 'r.', 'MarkerSize', 10);
% % title('origin dataset');
% 
% %pause;
% 
% % set k clusters, and init method
% k = 4;
% init = 'euclidean';
% 
% % % init by hand
% init_centers = zeros(k,d);
% % init_centers(1,:)=[0.04607 4.565];
% % init_centers(2,:)=[-3.983 -3.781];
% % init_centers(3,:)=[2.993 -2.907];
% 
% % init_centers(1,:)=[2.0188 3.574];
% % init_centers(2,:)=[2.181 4.6575];
% % init_centers(3,:)=[3.5908 4.5265];
% 
% % clustering
% [cluster_i,~,init_c] = k_means_alg(twoD,k,init,init_centers);
% 
% % plot clustering in different color
% colors = {'green', 'blue','black','red','magenta','yellow','cyan'};
% %colors = zeros(k,3);
% % % set color
% % for i = 1:k
% %     colors(i,:) = rand(1,3);
% % end
% 
% figure(2)
% %subplot(1,2,2);
% % if inti by hand plot init center
% % if isequal(init,'manual')
% %     plot(init_centers(:,1), init_centers(:,2), 'r*', 'MarkerSize', 12);
% %     hold on
% % end
% plot(init_c(:,1), init_c(:,2), 'r*', 'MarkerSize', 12);
% hold on
% for i = 1:k
%     hold on;
%     %plot(x(find(cluster_i==i,N)),t(find(cluster_i==i,N)),'.','Color',colors(i,:),'MarkerSize', 10);
%     plot(x(find(cluster_i==i),1),t(find(cluster_i==i),1),'.','color',colors{i},'MarkerSize', 10);
% end
% title('clustering');
% fprintf('finish! plz enter!')
% pause;


% % (d) run the algorithm for k = 1, . . . 10 and plot the k-means cost
% 
% clc;clear;close all;
% % load data
% twoD=load('twodpoints.txt');
% [N, d] = size(twoD);
% x = twoD(:,1);
% t = twoD(:,2);
% 
% % init
% k = 1:10;
% init = 'euclidean';
% costs = zeros(length(k),1);
% 
% for i = k
%     init_centers = zeros(i,d);
%     [~,cost] = k_means_alg(twoD,i,init,init_centers);
%     costs(i,:) = cost;
%     
% end
% plot(costs)



% % (e) load 'threedpoints.txt' and plot
% clc;clear;close all;
% % load data
% threeD=load('threedpoints.txt');
% [N, d] = size(threeD);
% 
% % init
% k = 1:10;
% init = 'euclidean';
% costs = zeros(length(k),1);
% 
% for i = k
%     init_centers = zeros(i,d);
%     [~,cost] = k_means_alg(threeD,i,init,init_centers);
%     costs(i,:) = cost;
%     
% end
% plot(costs)
% 
% % load data
% threeD=load('threedpoints.txt');
% [N, d] = size(threeD);
% x = threeD(:,1);
% y = threeD(:,2);
% z = threeD(:,3);
% 
% 
% % plot origin dataset
% figure(1);
% %subplot(1,2,1);
% plot3(x, y, z,'r.', 'MarkerSize', 10);
% title('origin dataset');
% grid on
% 
% %pause;
% 
% % set k clusters, and init method
% k = 4;
% init = 'euclidean';
% 
% % clustering
% [cluster_i,~, init_c] = k_means_alg(threeD,k,init,init_centers);
% 
% % plot clustering in different color
% colors = {'green', 'blue','black','red','cyan','magenta','yellow'};
% %colors = zeros(k,3);
% % % set color
% % for i = 1:k
% %     colors(i,:) = rand(1,3);
% % end
% 
% figure(2)
% plot3(init_c(:,1), init_c(:,2), init_c(:,3), 'r*', 'MarkerSize', 12);
% hold on
% k
% for i = 1:k
%     colors{i}
%     hold on;
%     %plot(x(find(cluster_i==i,N)),t(find(cluster_i==i,N)),'.','Color',colors(i,:),'MarkerSize', 10);
%     plot3(x(find(cluster_i==i),1),y(find(cluster_i==i),1), z(find(cluster_i==i),1),'.','color',colors{i},'MarkerSize', 10);
% end
% grid on
% title('clustering');
% fprintf('finish! plz enter!')
% pause;

% % (f) Load the UCI ?seeds? dataset from the last assignment and repeat the above step.
% 
% clc;clear;close all;
% % load data
% seedD=load('seeds_dataset.txt');
% [~, d] = size(seedD);
% seedD=seedD(:,1:d-1);
% [N, d] = size(seedD);
% 
% % init
% k = 1:10;
% init = 'euclidean';
% costs = zeros(length(k),1);
% 
% for i = k
%     init_centers = zeros(i,d);
%     [~,cost] = k_means_alg(seedD,i,init,init_centers);
%     costs(i,:) = cost;
%     
% end
% plot(costs)

% % load data
% clc;clear;close all;
% seedD=load('seeds_dataset.txt');
% [~, d] = size(seedD);
% % % init centers
% % init_centers = zeros(3,d-1);
% % for i = 1:3
% %     D=seedD((seedD(:,d)==i),:);
% %     init_centers(i,:) = D(unidrnd(70),1:d-1);
% % end
% % remove last col
% label=seedD(:,d);
% seedD=seedD(:,1:d-1);
% [N, d] = size(seedD);
% % set k clusters, and init method
% k = 3;
% init = 'euclidean';
% 
% 
% 
% 
% % init by hand
% init_centers = zeros(k,d);
% % init_centers(1,:)=[15.26 14.84 0.871 5.763 3.312 2.221 5.22];
% % init_centers(2,:)=[16.84 15.67 0.8623 5.998 3.484 4.675 5.877];
% % init_centers(3,:)=[11.21 13.13 0.8167 5.279 2.687 6.169 5.275];
% 
% min_loss = realmax;
% for i = 1:500
%     % clustering
%     [cluster_i,~, ~] = k_means_alg(seedD,k,init,0);
%     Diff = label ~= cluster_i;
%     loss = sum(Diff);
%     
%     if loss < min_loss
%         min_loss = loss;
%         opt_classifier = cluster_i;
%     end
% end
% min_loss
% opt_classifier;

clc;clear;close all;

x = [1 1 5 3];
t = [1 2 1 2];
D = [x',t'];
D
plot(x,t,'r.','MarkerSize', 20)
% set k clusters, and init method
k = 2;
init = 'uniform';

% clustering
[cluster_i,~,init_c] = k_means_alg(D,k,init,0);

% plot clustering in different color
colors = {'blue','red'};

figure(2)
plot(init_c(:,1), init_c(:,2), 'r*', 'MarkerSize', 12);
hold on
for i = 1:k
    hold on;
    plot(x(find(cluster_i==i)),t(find(cluster_i==i)),'.','color',colors{i},'MarkerSize', 20);
end
title('clustering');




