% cublasGetVector - Wrapper to CUBLAS cublasGetVector function
% 
% DESCRIPTION
% Wrapper to CUBLAS cublasGetVector function. Original function
% declaration:
% 
% cublasStatus
% cublasGetVector
% (int n, int elemSize, const void *x, int incx,
%  void *y, int incy)
% 
% 
% EXAMPLE
% 
% A = GPUsingle([1 2 3 4]);
% 
% % Ah should be of the correct type. GPUsingle is single
% % precision floating point, also Ah should be single
% % precision
% Ah = single(zeros(size(A)));
% 
% % The function getSizeOf returns the size of the
% % stored elements in A (for example float or complex)
% [status Ah] = cublasGetVector (numel(A), getSizeOf(A),...
%                     getPtr(A), 1, Ah, 1);
% cublasCheckStatus( status, 'Error.');
% disp(Ah);
