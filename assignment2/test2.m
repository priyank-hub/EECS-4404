% load the dataset and spilt
D = load('seeds_dataset.txt');



[D_1, D_2, D_3] = spiltDateset(D);
[w_1,hi_loss_1,bi_loss_1] = soft_svm(D_1, 210, 0.001, 0);
[w_2,hi_loss_2,bi_loss_2] = soft_svm(D_2, 210, 0.001, 0);
[w_3,hi_loss_3,bi_loss_3] = soft_svm(D_3, 210, 0.001, 0);


w_1
w_2
w_3