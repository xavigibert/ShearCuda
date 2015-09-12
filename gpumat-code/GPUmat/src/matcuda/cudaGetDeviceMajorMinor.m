% cudaGetDeviceMajorMinor - Returns CUDA compute capability
% major and minor numbers.
% 
% MODULE NAME
% na
% 
% DESCRIPTION
% Returns CUDA compute capability major and minor numbers.
% [STATUS, MAJOR, MINOR] = cudaGetDeviceMajorMinor(DEV)
% returns the compute capability number (major, minor) of the
% device=DEV. STATUS is the result of the operation.
% 
% EXAMPLE
% 
% dev = 0;
% [status,major,minor] = cudaGetDeviceMajorMinor(dev);
% if (status ~=0)
%   error(['Unable to get the compute capability']);
% end
% 
% major
% minor
