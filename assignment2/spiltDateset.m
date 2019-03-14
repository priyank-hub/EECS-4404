 function [D_1, D_2, D_3] = spiltDateset(D)
 
% % each spilt D into two part {-1,1}
% D_1 = D;
% D_1(D(:,8)==1, 8) = 1;
% D_1(D(:,8)~=1, 8) = -1;
% 
% D_2 = D;
% D_2(D(:,8)==2, 8) = 1;
% D_2(D(:,8)~=2, 8) = -1;
% 
% D_3 = D;
% D_3(D(:,8)==3, 8) = 1;
% D_3(D(:,8)~=3, 8) = -1;  
[~,C] = size(D);

% each spilt D into two part {-1,1}
D_1 = D;
D_1(D(:,C)==1, C) = 1;
D_1(D(:,C)~=1, C) = -1;

D_2 = D;
D_2(D(:,C)==2, C) = 1;
D_2(D(:,C)~=2, C) = -1;

D_3 = D;
D_3(D(:,C)==3, C) = 1;
D_3(D(:,C)~=3, C) = -1;  