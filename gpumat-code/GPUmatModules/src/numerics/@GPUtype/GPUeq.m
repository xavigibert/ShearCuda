% GPUeq - Equal
% 
% SYNTAX
% 
% GPUeq(X,Y,R)
% X - GPUsingle, GPUdouble
% Y - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUeq(A, B, R) is equivalent to eq(A, B), but result is returned
% in the input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% A = GPUsingle([1 2 0 4]);
% B = GPUsingle([1 0 0 4]);
% R = zeros(size(A), GPUsingle);
% GPUeq(A, B, R);
