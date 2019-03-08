function v = hinge_loss(w, x, t)
%clc;clear
% x=[1 2 3; 4 5 6; 7 8 9]
% t=[1;1;1]
% w=[1 7 8; 4 9 0; 7 8 9]
if 1 - t * dot(w, x) <= 0
    v = 0;
elseif 1 - t * dot(w, x) > 0
    v = -t*x;
end
% 
% a=x(1,:)
% b=w(1,:)
% c=dot(x(1,:), w(1,:))