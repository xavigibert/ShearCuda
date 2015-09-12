function GPUcompileCheck
% GPUcompileCheck - Checks if the compiler is configured 
% 
% SYNTAX
% 
% GPUcompileCheck
% 
% 

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


mypath = mfilename('fullpath');
mypath = mypath(1:end-length(mfilename)-1-4);

% load nvidia settings (CUDA_ROOT, CC_BIN)
nv = GPUmatGetCUDAInfo;
CUDA_ROOT = nv.CUDA_ROOT;

% check if all libraries are available
if (nv.cublasdll==0)
  error('Please check your CUDA installation. The CUBLAS dynamic library cannot be located on your computer.');
end

if (nv.cudartdll==0)
  error('Please check your CUDA installation. The RUNTIME dynamic library cannot be located on your computer.');
end

if (nv.cufftdll==0)
  error('Please check your CUDA installation. The CUFFT dynamic library cannot be located on your computer.');
end

file = fullfile(mypath,'util','testcompile.cpp');
GPUcompileMEX({file}); 

