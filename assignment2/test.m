clc;clear

w=[1 2 3 4]
x=[7 8 9 10]
t=1
if 1 - t * dot(w, x) <= 0
    v = 0;
elseif 1 - t * dot(w, x) > 0
    v = -t*x;
end