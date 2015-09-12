% cufftPlan3d - Wrapper to CUFFT cufftPlan3d function
% 
% DESCRIPTION
% Wrapper to CUFFT cufftPlan3d function. Original function decla-
% ration:
% 
% cufftResult
% cufftPlan2d(cufftHandle *plan,
%              int nx, int ny, int nz,
%              cufftType type);
% 
% Original function returns only a cufftResult, whereas wrapper re-
% turns also the plan.
