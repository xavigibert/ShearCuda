% setReal - Set a GPU variable as real
% 
% SYNTAX
% 
% setReal(A)
% A - GPU variable
% 
% 
% MODULE NAME
% na
% 
% DESCRIPTION
% setReal(P) sets the GPU variable P as real. Should be called
% before using GPUallocVector.
% 
% EXAMPLE
% 
% A = GPUsingle();
% setSize(A,[10 10]);
% setReal(A);
% GPUallocVector(A);
