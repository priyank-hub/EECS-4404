clc;clear;close all;

% STEP1 load data
[x, t] = load_data;

% STEP2 compute the empirical square loss on the data 
% and plot it as a function of W
W = [1:20];
p = polyfit(x,t,1);