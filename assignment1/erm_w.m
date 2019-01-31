function w = erm_w(x, t, d)
x = load('dataset1_inputs.txt');
t = load('dataset1_outputs.txt');
N = size(x, 1);
% design matrix X of the data, 
% where N is size of the data and d is degree of poly
% |x1^0 x1^1 ... x1^d|
% |x2^0 X2^1 ... X2^d|
% |...               | = X
% |...               |
% |xN^0 xN^1 ... xN^d|
X = zeros(N,d);
for r = 1:N
    for c = 1:d
        X(r, c) = x(r)^c;
    end
end
% first column would be constant
X = [ones(N,1), X];

% vector w that solves the unregularized least squares linear regression problem
% ERM solution w = (X'*X)^-1 * X' * t from slide,
% where X is design matrix of the data
w = (X' * X)^-1 * X' * t;
