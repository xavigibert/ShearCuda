% getSizeOf - Get the size of the GPU datatype (similar to sizeof in
% C)
% 
% SYNTAX
% 
% R = getSizeOf(X)
% X - GPU variable
% R - the size of the GPU variable datatype
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% This is a low level function used to get the size of the datatype of
% the GPU variable.
% Compilation not supported
% 
% EXAMPLE
% 
% A = rand(10,GPUsingle);
% getSizeOf(A)
