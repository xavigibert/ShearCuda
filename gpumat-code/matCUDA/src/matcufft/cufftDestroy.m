% cufftDestroy - Wrapper to CUFFT cufftDestroy function
% 
% DESCRIPTION
% Wrapper to CUFFT cufftDestroy function. Original function dec-
% laration:
% 
% cufftResult
% cufftDestroy(cufftHandle plan);
% 
% 
% EXAMPLE
% 
% fftType = cufftType;
% I = sqrt(-1);
% A = GPUsingle(rand(1,128)+I*rand(1,128));
% plan = 0;
% type = fftType.CUFFT_C2C;
% [status, plan] = cufftPlan1d(plan, numel(A), type, 1);
% cufftCheckStatus(status, 'Error in cufftPlan1D');
% 
% [status] = cufftDestroy(plan);
% cufftCheckStatus(status, 'Error in cuffDestroyPlan');
