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
fold = 2;
x = [1:10];
t = [21:30];
x=x';
t=t';
rank_data = cross(x,t);
for i = 1:fold
    k=1
    for j = 1+(i-1)*fold:i*size(rank_data,1)/fold
        a(k,:) = rank_data(j,:)
        k=k+1
    end
end
a

