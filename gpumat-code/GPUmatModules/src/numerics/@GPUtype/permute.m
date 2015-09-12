% permute - Permute array dimensions
% 
% SYNTAX
% 
% R   =   permute(X,   ORDER)
% X   -   GPUsingle,   GPUdouble
% Y   -   GPUsingle,   GPUdouble
% R   -   GPUsingle,   GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% R = PERMUTE(X,ORDER) rearranges the dimensions of X so that the-
% yare in the order specified by the vector ORDER.
% Compilation supported
% 
% EXAMPLE
% 
% A = rand(3,4,5,GPUsingle);
% B = permute(A,[3 2 1]);
