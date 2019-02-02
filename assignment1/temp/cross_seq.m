function part = cross_seq(x,fold)
c = zeros(size(x,1)/fold,fold);
for i = 1:fold
     c(:,i) = x((1+(i-1)*fold):fold*i,1);
end
part = c;