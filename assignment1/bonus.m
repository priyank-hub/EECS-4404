% STEP 6 bonus
clc;clear;close all
x_b = load("dataset2_inputs.txt");
t_b = load("dataset2_outputs.txt");
interval = -1:0.01:1.25;

[opt_w, min_loss,flag] = train_6(x_b,t_b);
for i = 1:5
    [w,l] = train_6(x_b,t_b);
    if l < min_loss 
        min_loss = l;
        opt_w = w;
    end
end

opt_w

flag


p = polyfit(x_b,t_b,7);
plot(interval,polyval(p,interval));
hold on

plot(interval,func(opt_w,interval));
hold on

opt_w_a = [0.6006;1.1576;-11.8218;9.7903;33.6301;-27.3195;-22.0138;16.8897];
plot(interval,func(opt_w_a,interval));

legend("fit","w","w_a");

plot(x_b,t_b,'rx');




title('Visualization Model vs OriginDataPoints');
ylabel('outputs t');
xlabel('inputs x');
save('opt_w_s6.txt','opt_w');
fprintf("step6: Finish training model, please press enter to continue!\n")
pause;
clc;close all;