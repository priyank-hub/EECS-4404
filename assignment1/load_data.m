function [x, t] = load_data(a)
% Load Data
% input = load('dataset1_inputs.txt');
% output = load('dataset1_outputs.txt');
input = load(a(1));
output = load(a(2));
x = input(:, 1); t = output(:, 1);
m = length(t); % number of training examples

% Plot Data
fprintf('Plotting Data ...\n');
plot(x, t, 'rx', 'MarkerSize', 10);
title('Dataset1 Plot');
ylabel('dataset1_outputs');
xlabel('dataset1_inputs');