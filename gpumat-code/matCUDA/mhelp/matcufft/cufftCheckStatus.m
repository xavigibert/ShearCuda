% cufftCheckStatus - Checks the CUFFT status
% 
% DESCRIPTION
% cufftCheckStatus(STATUS,MSG) returns EXIT FAILURE(1) or
% EXIT SUCCESS(0) depending on STATUS value, and throws an er-
% ror with message 'MSG'. STATUS is compared to CUFFT possible
% results.
% 
% EXAMPLE
% 
% fftType = cufftType;
% A = GPUsingle(rand(1,128));
% plan = 0;
% type = fftType.CUFFT_C2C;
% [status, plan] = cufftPlan1d(plan, numel(A), type, 1);
% cufftCheckStatus(status, 'Error in cufftPlan1D');
