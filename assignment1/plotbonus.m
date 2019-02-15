%w1=load("bonus.txt");
w2=load("bonus2.txt");
%w3=load("bonus3.txt");
w4=load("bonus4.txt");
w5=load("bonus5.txt");

x_b = load("dataset2_inputs.txt");
t_b = load("dataset2_outputs.txt");
interval = -1:0.01:1.25;
hold on
%plot(interval,func(w1,interval));
plot(interval,func(w2,interval));
%plot(interval,func(w3,interval));
plot(interval,func(w4,interval));
plot(interval,func(w5,interval));


p = polyfit(x_b,t_b,7);
plot(interval,polyval(p,interval));

legend("w2","w4","w5","p");


plot(x_b,t_b,'rx');


