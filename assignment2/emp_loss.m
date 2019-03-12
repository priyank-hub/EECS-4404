function emr_loss = emp_loss(w, D, flag)
% L_D(w)=(1/N)sum(loss(w,(x_i, t_i)))
[N,clo] = size(D);
l=0;
for i = 1:N
    x_i = D(i,1:clo-1);
    t_i = D(i,clo);
    if(strcmpi(flag, 'hinge'))
        l = l + hinge_loss(w, x_i, t_i);
    elseif(strcmpi(flag, 'binary'))
        l = l + binary_loss(w, x_i, t_i);
    end
end    
emr_loss = (1/N) * l;