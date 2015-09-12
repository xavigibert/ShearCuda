% setComplex - Set a GPU variable as complex
% 
% SYNTAX
% 
% setComplex(A)
% A - GPU variable
% 
% 
% MODULE NAME
% na
% 
% DESCRIPTION
% setComplex(P) set the GPU variable P as complex. Should be
% called before using GPUallocVector.
% 
% EXAMPLE
% 
% A = GPUsingle();
% setSize(A,[10 10]);
% setComplex(A);
% GPUallocVector(A);
