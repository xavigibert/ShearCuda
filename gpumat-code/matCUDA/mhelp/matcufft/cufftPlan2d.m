% cufftPlan2d - Wrapper to CUFFT cufftPlan2d function
% 
% DESCRIPTION
% Wrapper to CUFFT cufftPlan2d function. Original function decla-
% ration:
% 
% cufftResult
% cufftPlan2d(cufftHandle *plan,
%              int nx, int ny,
%              cufftType type);
% 
% 
% EXAMPLE
% 
% fftType = cufftType;
% I = sqrt(-1);
% A = GPUsingle(rand(128,128)+I*rand(128,128));
% plan = 0;
% % Vectors stored in column major format (FORTRAN)
% s = size(A);
% type = fftType.CUFFT_C2C;
% [status, plan] = cufftPlan2d(plan, s(2), s(1),type);
% cufftCheckStatus(status, 'Error in cufftPlan2D');
% 
% [status] = cufftDestroy(plan);
% cufftCheckStatus(status, 'Error in cuffDestroyPlan');
