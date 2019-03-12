clc;clear;close all;

% PART (a) Implement SGD for SoftSVM
dataset = load_data("bg.txt");
num_updates = 500;
stepsize_sequence = 0.01;
n = 50;

% skip the averaging over weight vectors for
% the output and instead, simply output the last iterate
% if n=0, we can skip the  the averaging over weight 
% vectors in soft SVM iteration
% output weight `w`, empiral hinge loss and empiral binary loss
[w,hi_loss,bi_loss] = soft_svm(dataset, num_updates, 1/stepsize_sequence, n);

% PART (b) Run with lambda = {100, 10, 1, .1, .01, .001} on the data
% and print empiral hinge loss and empiral binary loss
clc;clear;close all;
dataset = load_data("bg.txt");
num_updates = 300;
s_lambda = [100, 10, 1, .1, .01, .001];
n = 1;

for i = 1:length(s_lambda)
    [w,hi_loss,bi_loss] = soft_svm(dataset, num_updates, s_lambda(i), n);
  
    figure
    title(['EMP over lambda = ', num2str(s_lambda(i))])
    hold on
    set(gca, 'YScale', 'log'); % set y-axis as log
    plot(hi_loss, 'Color', 'red');
    plot(bi_loss, 'Color', 'blue');
        
    legend('hinge loss','binary loss');
    hold off;
    xlabel('Iteration'); 
    ylabel(['EMP Loss of w = ', num2str(w)]);
end

fprintf("part(b): Finished! Please press enter to continue!\n")
pause;
clc;close all;

% PART (d-e) Spilt the data set and train three binary linear predictors
clc;clear;close all;

% load the dataset and spilt
D = load('seeds_dataset.txt');
[D_1, D_2, D_3] = spiltDateset(D);





