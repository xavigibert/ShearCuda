function ret = cublasCheckStatus(status,message)
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

EXIT_FAILURE	= 1;
EXIT_SUCCESS	= 0;
CUBLAS_STATUS_SUCCESS = 0;

ret = EXIT_SUCCESS;

if (status ~= CUBLAS_STATUS_SUCCESS)
    ret = EXIT_FAILURE;
    error(message);
end

