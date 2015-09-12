% cublasGetError - Wrapper to CUBLAS cublasGetError function
% 
% DESCRIPTION
% Wrapper to CUBLAS cublasGetError function. Original function
% declaration:
% 
% cublasStatus
% cublasGetError (void)
% 
% 
% EXAMPLE
% 
% status = cublasGetError();
% cublasCheckStatus( status, 'Kernel execution error');
