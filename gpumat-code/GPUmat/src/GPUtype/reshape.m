% reshape - Reshape array
% 
% SYNTAX
% 
% R   =   reshape(X,m,n)
% R   =   reshape(X,m,n,p,...)
% R   =   reshape(X,[m n p ...])
% X   -   GPUsingle, GPUdouble
% R   -   GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% na
% 
% DESCRIPTION
% R = reshape(X,m,n) returns the m-by-n matrix R whose elements
% are taken column-wise from X.
% R = reshape(X,m,n,p,...) or B = reshape(A,[m n p ...])
% returns an n-dimensional array with the same elements as X but
% reshaped to have the size m-by-n-by-p-by-....
% 
% EXAMPLE
% 
% X = rand(30,1,GPUsingle);
% R = reshape(X, 6, 5);
% R = reshape(X, [6 5]);
