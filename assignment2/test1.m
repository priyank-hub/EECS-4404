D = load('seeds_dataset.txt');
% + bias
[N,C] = size(D);
bias = ones(N,1);
D = [D(:,1:C-1) bias D(:,C)];
[N,C] = size(D);
ws=opt_ws;
DR = zeros(N,1);
    for j = 1:N
        x_j = D(j, 1:C-1);
        t_j = D(j,C);
        [~, I] = max([dot(ws(1,:),x_j), dot(ws(2,:),x_j), dot(ws(3,:),x_j)]);
        DR(j,1) = I;
    end
    
sum(DR ~= D(:,C))