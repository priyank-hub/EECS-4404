clc;clear;close all
% (b)-(c) load 'twodpoints.txt' and plot

% load data
twoD=load('twodpoints.txt');
[N, d] = size(twoD);
x = twoD(:,1);
t = twoD(:,2);

% plot origin dataset
figure(1);
%subplot(1,2,1);
plot(x, t, 'r.', 'MarkerSize', 10);
title('origin dataset');

%pause;

% set k clusters, and init method
k = 3;
init = 'uniform';

% init by hand
init_centers = zeros(k,d);
init_centers(1,:)=[0.04607 4.565];
init_centers(2,:)=[-3.983 -3.781];
init_centers(3,:)=[2.993 -2.907];

% clustering
[cluster_i,~] = k_means_alg(twoD,k,init,init_centers);

% plot clustering in different color
%colors = {'green', 'blue','black','yellow','red','magenta','cyan'};
colors = zeros(k,3);
% set color
for i = 1:k
    colors(i,:) = rand(1,3);
end

figure(2)
%subplot(1,2,2);
% if inti by hand plot init center
if isequal(init,'manual')
    plot(init_centers(:,1), init_centers(:,2), 'r*', 'MarkerSize', 12);
    hold on
end
for i = 1:k
    hold on;
    plot(x(find(cluster_i==i,N)),t(find(cluster_i==i,N)),'.','Color',colors(i,:),'MarkerSize', 10);
end
title('clustering');
fprintf('finish! plz enter!')
pause;


% (d) run the algorithm for k = 1, . . . 10 and plot the k-means cost

clc;clear;close all;
% load data
twoD=load('twodpoints.txt');
[N, d] = size(twoD);
x = twoD(:,1);
t = twoD(:,2);

% init
k = 1:10;
init = 'uniform';
costs = zeros(length(k),1);

for i = k
    init_centers = zeros(i,d);
    [~,cost] = k_means_alg(twoD,i,init,init_centers);
    costs(i,:) = cost;
    
end
plot(costs)



% (e) load 'threedpoints.txt' and plot







