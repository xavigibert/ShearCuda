% GPUallocVector - Variable allocation on GPU memory
% 
% SYNTAX
% 
% GPUallocVector(P)
% P - GPU variable
% 
% 
% MODULE NAME
% na
% 
% DESCRIPTION
% P = GPUallocVector(P) allocates the required GPU memory for
% P. The size of the allocated variable depends on the size of P.
% A complex variable is allocated as an interleaved sequence of real
% and imaginary values. It means that the memory size for a complex
% on the GPU is numel(P)*2*SIZE OF FLOAT. It is mandatory to set
% the size of the variable before calling GPUallocVector.
% 
% EXAMPLE
% 
% A = GPUsingle();
% setSize(A,[100 100]);
% GPUallocVector(A);
% 
% A = GPUsingle();
% setSize(A,[100 100]);
% setComplex(A);
% GPUallocVector(A);
