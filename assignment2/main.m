clc;clear;close all;

% PART (a) Implement SGD for SoftSVM
% dataset = load_data("bg.txt");
% num_updates = 500;
% stepsize_sequence = 0.01;
% n = 50;

% skip the averaging over weight vectors for
% the output and instead, simply output the last iterate
% if n=0, we can skip the  the averaging over weight 
% vectors in soft SVM iteration
% output weight `w`, empiral hinge loss and empiral binary loss
% [w,hi_loss,bi_loss] = soft_svm(dataset, num_updates, 1/stepsize_sequence, n);

% PART (b) Run with lambda = {100, 10, 1, .1, .01, .001} on the data
% and print empiral hinge loss and empiral binary loss
% clc;clear;close all;
% dataset = load_data("bg.txt");
% num_updates = 300;
% s_lambda = [100, 10, 1, .1, .01, .001];
% n = 0;
% 
% for i = 1:length(s_lambda)
%     [w,hi_loss,bi_loss] = soft_svm(dataset, num_updates, s_lambda(i), n);
%   
%     figure
%     title(['EMP Hinge Loss over lambda = ', num2str(s_lambda(i))])
%     hold on
% %     set(gca, 'YScale', 'log'); % set y-axis as log
%     plot(hi_loss, 'Color', 'red'); 
%     xlabel('Iteration'); 
%     ylabel('Hinge Loss');
%     hold off
%     
%     figure
%     title(['EMP Binary Loss over lambda = ', num2str(s_lambda(i))])
%     hold on
% %     set(gca, 'YScale', 'log'); % set y-axis as log
%     plot(bi_loss, 'Color', 'blue');
%     xlabel('Iteration'); 
%     ylabel('EMP Binary Loss');
%     hold off
% end
% 
% fprintf("part(b): Finished! Please press enter to continue!\n")
% pause;
% clc;close all;

% PART (d-e) Spilt the data set and train three binary linear predictors
clc;clear;close all;

% load the dataset and spilt
D = load('seeds_dataset.txt');
[N,C] = size(D);
[D_1, D_2, D_3] = spiltDateset(D);
num_updates = 220;
lambda = 100;
binary_losses=zeros(3,1);
ws = zeros(3,C-1);


for i = 1:20
    
    [w_1,hi_loss_1,bi_loss_1] = soft_svm(D_1, num_updates, lambda, 0);
%     binary_losses(1) = emp_loss(w_1, D_1, 'binary');
    ws(1,:) = w_1;
    [w_2,hi_loss_2,bi_loss_2] = soft_svm(D_2, num_updates, lambda, 0);
%     binary_losses(2) = emp_loss(w_2, D_2, 'binary');
    ws(2,:) = w_2;
    [w_3,hi_loss_3,bi_loss_3] = soft_svm(D_3, num_updates, lambda, 0);
%     binary_losses(3) = emp_loss(w_3, D_3, 'binary');
    ws(3,:) = w_3;
    
    min_loss = N;
    
    DR = D(:,C);
    for j = 1:N
        x_j = D(j, 1:C-1);
        t_j = D(j,C);
        [~, I] = max([dot(ws(1,:),x_j), dot(ws(2,:),x_j), dot(ws(3,:),x_j)]);
        DR(j,1) = I;
    end
    Diff = DR(:,1) ~= D(:,C);
    loss = sum(Diff)
    
    if (loss < min_loss)
        min_loss = loss
        opt_DR = DR;
        opt_ws = ws;
    end
    
    i
    
    
end
    
min_loss;    
%     
%     figure
%     title("bi_loss on w_1")
%     hold on
%     plot(bi_loss_1, 'Color', 'red'); 
%     xlabel('Iteration'); 
%     ylabel('Binary Loss');
%     hold off
%     
%     figure
%     title("bi_loss on w_2")
%     hold on
%     plot(bi_loss_2, 'Color', 'blue'); 
%     xlabel('Iteration'); 
%     ylabel('Binary Loss');
%     hold off
%     
%     figure
%     title("bi_loss on w_3")
%     hold on
%     plot(bi_loss_3, 'Color', 'green'); 
%     xlabel('Iteration'); 
%     ylabel('Binary Loss');
%     hold off
% pause;
% clc;close all;



