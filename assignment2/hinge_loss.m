function h_loss = hinge_loss(w, x_i, t_i)
% hinge_loss(w,(x,t)) = max{0, 1-t<w,x>}
if t_i * dot(w,x_i) > 1
    h_loss = 0;
else
    h_loss = 1 - t_i * dot(w,x_i);
end
