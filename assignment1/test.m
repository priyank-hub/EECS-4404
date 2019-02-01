clc;clear;close all;
x = load('dataset1_inputs.txt');
t = load('dataset1_outputs.txt');
fold = 10;
% cross = zeros(size(x,1)/fold,fold)
% % cross(:,1) = x(1:10,1)
% % cross(:,2) = x(11:10+10,1)
% 
%  for i = 1:fold
%      cross(:,i) = x((1+(i-1)*fold):fold*i,1)
%  end
fold = 5;
x = [1:10];
t = [21:30];
x=x';
t=t';
% rank_data = cross(x,t);
%     n = 1
%     m = 1
% for i = 1:10
% 
%     for j = 1+(i-1)*fold:i*size(rank_data,1)/fold
%         if n < 2*size(rank_data,1)/fold
%             training(n,:) = rank_data(j,:);
%             n = n + 1;
%         else
%             testing(m,:) = rank_data(j,:);
%             m = m + 1;
%         end
%     end
% end
% testing
% training
% n = 1;m = 1;
% chunck = size(rank_data,1)/fold;
% for i = 1:size(rank_data,1)
%     if i <= (fold-1) * chunck
%         training(n,:) = rank_data(i,:);
%         n = n + 1;
%     else
%         testing(m,:) = rank_data(i,:);
%         m = m + 1;
%     end
% end
% training
% testing
[tr, te] = cross(x,t,fold);
tr
te