% subsref - Subscripted reference
% 
% SYNTAX
% 
% R   =   X(I)
% X   -   GPUsingle, GPUdouble
% I   -   GPUsingle, GPUdouble, Matlab range
% R   -   GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% A(I) (subsref) is an array formed from the elements of A specified
% by the subscript vector I. The resulting array is the same size as
% I except for the special case where A and I are both vectors. In
% this case, A(I) has the same number of elements as I but has the
% orientation of A.
% Compilation not supported
% 
% EXAMPLE
% 
% A =     GPUsingle([1 2 3 4 5]);
% A =     GPUdouble([1 2 3 4 5]);
% idx     = GPUsingle([1 2]);
% B =     A(idx)
