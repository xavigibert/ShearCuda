% getPtr - Get pointer on GPU memory
% 
% SYNTAX
% 
% R = getPtr(X)
% X - GPU variable
% R - the pointer to the GPU memory region
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% This is a low level function used to get the pointer value to the
% GPU memory of a GPU variable
% 
% Compilation not supported
% 
% EXAMPLE
% 
% A = rand(10,GPUsingle);
% getPtr(A)
