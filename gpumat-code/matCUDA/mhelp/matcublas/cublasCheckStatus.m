% cublasCheckStatus - Check the CUBLAS status.
% 
% DESCRIPTION
% cublasCheckStatus(STATUS,MSG) returns EXIT FAILURE(1) or
% EXIT SUCCESS(0) depending on STATUS value, and throws an er-
% ror with message 'MSG'.
% 
% EXAMPLE
% 
% status = cublasGetError();
% cublasCheckStatus( status, 'Kernel execution error');
