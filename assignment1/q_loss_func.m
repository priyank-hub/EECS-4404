function Loss = q_loss_func(w, x, t)
N = size(x,1);
Loss = 0;
for i = 1:N
    Loss = Loss + 1/2 * (func(w,x(i)) - t(i))^2;
end
Loss = (1/2) * Loss;
