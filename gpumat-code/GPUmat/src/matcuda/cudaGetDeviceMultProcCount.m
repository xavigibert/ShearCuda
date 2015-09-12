% cudaGetDeviceMultProcCount - Returns device multi-processors
% count
% 
% MODULE NAME
% na
% 
% DESCRIPTION
% [STATUS, COUNT] = cudaGetDeviceMultProcCount(DEV)           re-
% turns the number of multi-processors of the device=DEV. STATUS
% is the result of the operation.
% 
% EXAMPLE
% 
% dev = 0;
% [status,count] = cudaGetDeviceMultProcCount(dev);
% if (status ~=0)
%   error('Error getting numer of multi proc');
% end
% disp(['    Mult. processors = ' num2str(count) ]);
