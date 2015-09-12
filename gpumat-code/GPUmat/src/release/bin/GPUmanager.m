function [y, v] = GPUmanager(varargin)

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

persistent GPUman
persistent GPUmanS
persistent GT
if (isempty(GPUman))
  gpuFLOAT   = 0;
  gpuCFLOAT  = 1;
  gpuDOUBLE  = 2;
  gpuCDOUBLE = 3;
  gpuINT32   = 4;

  % number of streams
  nstreams = varargin{1};
  filename = varargin{2};
  dev = varargin{3};
  major = varargin{4};
  minor = varargin{5};
  [GPUman, GPUmanS] = GPUmanagerCreate(nstreams,filename,dev,now,0, major, minor);

%   N = 5;
%   for i=0:(N-1)
%     index = i;
%     tmp = GPUsingle(index,index);
%     GT{index+1} = tmp;
%     
%   end
%   
%   for i=0:(N-1)
%     index = i+N;
%     tmp = GPUdouble(index,index);
%     GT{index+1} = tmp;
%   end
%   
%   GPUmatRegisterGPUtype(GPUman, 0,   N, GT, gpuFLOAT);
%   GPUmatRegisterGPUtype(GPUman, N, 2*N, GT, gpuDOUBLE);
  
   
else
    
end
mlock;

y = GPUman;
v = GPUmanS;


end


  
