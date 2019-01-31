function c = cross(x,y)
concat = horzcat(x,y);
rowrank = randperm(size(concat, 1));
c = concat(rowrank, :);