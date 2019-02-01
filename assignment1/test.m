clc;clear;close all;
x = load('dataset1_inputs.txt');
t = load('dataset1_outputs.txt');
% n=5
% legends = string(zeros(2*n,1))
% label = string(zeros(5,1));
% for i = [1 2 3 4 5]
%     label(i) = ("W = " +num2str(i)+" (ERM)")
% end
% label'
degrees = [1 5 10 20]
interval = -1.05:0.01:1.05;
n = 2*length(degrees)+1
label = string(zeros(n,1));