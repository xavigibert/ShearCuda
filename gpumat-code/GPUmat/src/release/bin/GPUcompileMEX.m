function GPUcompileMEX (infiles)
% GPUcompileMEX - Compiles mex files 
% 
% SYNTAX
% 
% GPUcompileMEX filenames
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

include = [' -I"' fullfile(mypath,'modules','include') '"'];

% Libraries and includes for CUDA
cudainclude = [' -I"' fullfile(CUDA_ROOT, 'include') '"'];
allinclude = [cudainclude ' ' include];
switch computer
  case {'PCWIN64'}
    libfolder = 'lib\x64';
  case {'PCWIN'}
    libfolder = 'lib\Win32';
  case {'GLNXA64'}
    libfolder = 'lib64';
  otherwise
    libfolder = 'lib';
end

lib = '';
cudalib  = [' -L"' fullfile(CUDA_ROOT, libfolder) '" -lcuda -lcudart -lcufft -lcublas'];
alllib = [cudalib ' ' lib];

% flags for mex compilation
flags = '';
if (isunix)
  mexflags = [flags ' -DUNIX'];
else
  mexflags = flags;
end





outdir = '.';
common   = fullfile(mypath,'modules','common','GPUmat.cpp');

%% Build
for i=1:length(infiles)
  file = [infiles{i} ' ' common];
  cmd = ['mex ' mexflags ' -outdir ' outdir ' ' file ' ' allinclude ' ' alllib ];
  disp(cmd);
  try
    eval(cmd);
  catch
    GPUcompileAbort
    err = lasterror;
    error(err.message);
  end
end

end

