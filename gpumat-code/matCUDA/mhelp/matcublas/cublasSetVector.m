% cublasSetVector - Wrapper to CUBLAS cublasSetVector function
% 
% DESCRIPTION
% Wrapper to CUBLAS cublasSetVector function. Original function
% declaration:
% 
% cublasStatus
% cublasSetVector
% (int n, int elemSize, const void *x, int incx,
%  void *y, int incy)
% 
% 
% EXAMPLE
% 
% B =single( [1 2 3 4]);
% 
% % Create empty GPU variable A
% A = GPUsingle();
% setSize(A, size(B));
% GPUallocVector(A);
% 
% status = cublasSetVector(numel(A), getSizeOf(A), ...
% B, 1, getPtr(A), 1);
% cublasCheckStatus( status, 'Error.');
% 
% disp(single(A));
