function emr_rl = emprical_hinge_loss(w, x, t, lambda)
% L_D(w)=(1/N)sum(hinge_loss(w,(x_i, t_i)))
N = size(x,1);
hl=0;
for i = 1:N
    hl = hl + hinge_loss(w, x(i,:), t(i));
end    
emr_rl = lambda * norm(w)^2 + (1/N) * hl;
