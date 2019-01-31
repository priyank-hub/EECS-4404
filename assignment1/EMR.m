function EMR()
x = load('dataset1_inputs.txt');
y = load('dataset1_outputs.txt');
for n=1:1
    ANSS=polyfit(x,y,n);  %?polyfit????
    for i=1:n+1           %answer??????????????????
       answer(i,n)=ANSS(i);
   end
    x0=0:0.01:17;
    y0=ANSS(1)*x0.^n    ; %??????????????????
    for num=2:1:n+1     
        y0=y0+ANSS(num)*x0.^(n+1-num);
    end
    %subplot(3,3,n)
    plot(x,y,'*', 'MarkerSize', 10)
    hold on
    plot(x0,y0)
end