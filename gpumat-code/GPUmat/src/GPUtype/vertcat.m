% vertcat - Vertical concatenation
% 
% SYNTAX
% 
% R   =   [X;Y]
% X   -   GPUsingle, GPUdouble
% Y   -   GPUsingle, GPUdouble
% R   -   GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% na
% 
% DESCRIPTION
% [A;B] is the vertical concatenation of matrices A and B. A and B
% must have the same number of columns. Any number of matrices
% can be concatenated within one pair of brackets.
% 
% EXAMPLE
% 
% A = [zeros(10,1,GPUsingle);colon(0,1,10,GPUsingle)'];
