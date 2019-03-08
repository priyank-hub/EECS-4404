function b_loss = binary_loss(w, x_i, t_i)
% binary_loss(w,(x,t)) = 0[y(x) = t]
if t_i * dot(w,x_i) >= 0
    b_loss = 0;
else
    b_loss = 1;
end