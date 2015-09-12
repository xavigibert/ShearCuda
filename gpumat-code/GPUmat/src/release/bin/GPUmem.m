function free_mem = GPUmem()
% GPUmem - Returns the free memory (bytes) on selected GPU
% device
% 
% SYNTAX
% 
% GPUmem
% 
% 
% DESCRIPTION
% Returns the free memory (bytes) on selected GPU device.
% 
% EXAMPLE
% 
% GPUmem
% GPUmem/1024/1024

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

% Clean up memory cache first
GPUmemClean;

free_mem = 0;
c = 0;
[status, free_mem, c] = cuMemGetInfo(free_mem, c);
cuCheckStatus(status,'Unable to retrieve information from GPU.');

end
