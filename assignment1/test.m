x = load('dataset1_inputs.txt');
t = load('dataset1_outputs.txt');
loss_rlm = zeros(20,1);
for i = 1:20
    w = rlm_w(x, t, 20, -i);
    loss_rlm(i) = q_loss(w, x, t);
end
loss_rlm