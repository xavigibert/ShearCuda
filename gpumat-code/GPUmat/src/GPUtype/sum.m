% sum - Sum of elements
% 
% SYNTAX
% 
% R =   sum(X)
% R =   sum(X, DIM)
% X -   GPUsingle, GPUdouble
% DIM   - integer
% R -   GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% na
% 
% DESCRIPTION
% S = SUM(X) is the sum of the elements of the vector X. S =
% SUM(X,DIM) sums along the dimension DIM.
% Note: currently the performance of the sum(X,DIM) with DIM>1 is
% 3x or 4x better than the sum(X,DIM) with DIM=1.
% 
% EXAMPLE
% 
% X = rand(5,5,GPUsingle)+i*rand(5,5,GPUsingle);
% R = sum(X);
% E = sum(X,2);
