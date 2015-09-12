% cufftExecC2C - Wrapper to CUFFT cufftExecC2C function
% 
% DESCRIPTION
% Wrapper to CUFFT cufftExecC2C function. Original function dec-
% laration:
% 
% cufftResult
% cufftExecC2C(cufftHandle plan,
%               cufftComplex *idata,
%               cufftComplex *odata,
%               int direction);
% 
% 
% EXAMPLE
% 
% fftType = cufftType;
% fftDir = cufftTransformDirections;
% 
% I = sqrt(-1);
% 
% A = GPUsingle(rand(1,128)+I*rand(1,128));
% plan = 0;
% type = fftType.CUFFT_C2C;
% [status, plan] = cufftPlan1d(plan, numel(A), type, 1);
% cufftCheckStatus(status, 'Error in cufftPlan1D');
% 
% dir = fftDir.CUFFT_FORWARD;
% [status] = cufftExecC2C(plan, getPtr(A),getPtr(A), dir);
% cufftCheckStatus(status, 'Error in cufftExecC2C');
% 
% [status] = cufftDestroy(plan);
% cufftCheckStatus(status, 'Error in cuffDestroyPlan');
