function ret = cufftCheckStatus(status,message)
%cufftCheckStatus  Check the CUFFT status.
%  cufftCheckStatus(STATUS,MSG) returns EXIT_FAILURE(1) or
%  EXIT_SUCCESS(0) depending on STATUS value, and throws an
%  error with message 'MSG'. STATUS is compared to CUFFT
%  possible results.

% globals

EXIT_FAILURE	= 1;
EXIT_SUCCESS	= 0;

ret = EXIT_SUCCESS;

if (status ~= 0)
    ret = EXIT_FAILURE;
    error([message '(' num2str(status) ')']);
end
