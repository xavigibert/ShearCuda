% cudaGetDeviceMemory - Returns device total memory
% 
% MODULE NAME
% na
% 
% DESCRIPTION
% [STATUS, TOTMEM] = cudaGetDeviceMemory(DEV) returns the to-
% tal memory of the device=DEV. STATUS is the result of the oper-
% ation.
% 
% EXAMPLE
% 
% dev = 0;
% [status,totmem] = cudaGetDeviceMemory(dev);
% if (status ~=0)
%   error('Error getting total memory');
% end
% totmem = totmem/1024/1024;
% disp(['Total memory=' num2str(totmem) 'MB']);
