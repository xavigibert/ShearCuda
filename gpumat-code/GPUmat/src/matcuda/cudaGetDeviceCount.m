% cudaGetDeviceCount - Wrapper to CUDA cudaGetDeviceCount
% function.
% 
% MODULE NAME
% na
% 
% DESCRIPTION
% Wrapper to CUDA cudaGetDeviceCount function.
% 
% EXAMPLE
% 
% count = 0;
% [status,count] = cudaGetDeviceCount(count);
% if (status ~=0)
%   error('Unable to get the number of devices');
% end
