function emr_rl = emprical_hinge_loss(w, x, t, lambda)
% x=[1 2 3; 4 5 6; 7 8 9]
% t=[1;1;1]
% w=[1 7 8; 4 9 0; 7 8 9]
N = size(x,1);
hl=0;

for i = 1:N
    hl = hl + hinge_loss(w, x(i,:), t(i));
end    
emr_rl = lambda * norm(w)^2 + (1/N) * hl;
