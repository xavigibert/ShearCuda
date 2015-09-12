function GPUdeviceInit(dev)
% GPUdeviceInit - Initializes a CUDA capable GPU device
% 
% SYNTAX
% 
% GPUdeviceInit(dev)
% dev - device number
% 
% 
% DESCRIPTION
% GPUdeviceInit(dev) initializes the GPU device dev, where dev is
% an integer corresponding to the device number. By using GPUinfo
% it is possible to see the available devices and the corresponding
% number
% 
% EXAMPLE
% 
% GPUinfo
% GPUdeviceInit(0)

%
%     Copyright (C) 2012  GP-you Group (http://gp-you.org)
% 
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <http://www.gnu.org/licenses/>.

if (GPUstart(1)==0)
  error('GPU not started. Please start the GPU using GPUstart');
end

count = -1;
[status,count] = cudaGetDeviceCount(count);
cudaCheckStatus(status,'Unable to get the number of devices');

if (count==0)
  error('cutil error: no devices supporting CUDA.');
end

if (dev>count)
  error('cutil error: specified devices doesn''t exist.');
end

[status,major,minor] = cudaGetDeviceMajorMinor(dev);
cudaCheckStatus(status,['Unable to get the compute capability for device N. ' num2str(dev) ]);

if (major<1)
  error('cutil error: device does not support CUDA.');
end


end

