clc;clear;close all;

% PART (a) Implement SGD for SoftSVM
dataset = load_data("bg.txt");
num_updates = 100;
stepsize_sequence = 0.1;

% skip the averaging over weight vectors for
% the output and instead, simply output the last iterate
[w,emp_loss,hi_loss] = soft_svm(dataset, num_updates, 1/stepsize_sequence);
plot(emp_loss)