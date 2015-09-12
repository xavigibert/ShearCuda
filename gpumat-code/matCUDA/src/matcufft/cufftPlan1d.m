% cufftPlan1d - Wrapper to CUFFT cufftPlan1d function
% 
% DESCRIPTION
% Wrapper to CUFFT cufftPlan1d function. Original function decla-
% ration:
% 
% cufftResult
% cufftPlan1d(cufftHandle *plan,
%              int nx,
%              cufftType type,
%              int batch);
% 
% Original function returns only a cufftResult, whereas wrapper re-
% turns also the plan.
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
