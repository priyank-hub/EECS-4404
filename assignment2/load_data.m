function dataset = load_data(a)
%load the data and add t_i clonum

x = load(a);
n = size(x,1);
t_1 = ones(n/2,1);
t_2 = - ones(n/2,1);
t = [t_1;t_2];
dataset = [x,t];
