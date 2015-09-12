function y = GPUmatGetCUDAInfo
%GPUmatGetCUDAInfo Return CUDA information

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

%%

% Check CUDA
cublasdll = 0;
cudartdll = 0;
cufftdll  = 0;

CUDA_ROOT = '';

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

archstr = computer('arch');

if (ispc)
  if (strcmp(archstr,'win32'))  
    cublasdyn = 'cublas32_*.dll';
    cudartdyn = 'cudart32_*.dll';
    cufftdyn  = 'cufft32_*.dll';
  else
    cublasdyn = 'cublas64_*.dll';
    cudartdyn = 'cudart64_*.dll';
    cufftdyn  = 'cufft64_*.dll';
  end
  cublaslib = 'cublas.lib';
  cudartlib = 'cudart.lib';
  cufftlib  = 'cufft.lib';
  
end

if (isunix)
  cublasdyn = 'libcublas.so';
  cudartdyn = 'libcudart.so';
  cufftdyn  = 'libcufft.so';
  
  cublaslib = 'libcublas.so';
  cudartlib = 'libcudart.so';
  cufftlib  = 'libcufft.so';
end

if (ispc)
  path = getenv('PATH');
end
if (isunix)
  path = getenv('LD_LIBRARY_PATH');
end

pathCell = splitPath(path);

for i=1:length(pathCell)
  pi = pathCell{i};
  % check for CUDA dlls
  if (existfile(fullfile(pi,cublasdyn)))
    if (existfile(fullfile(pi,'..',libfolder,cublaslib)))
      cublasdll = 1;
      CUDA_ROOT = fullfile(pi,'..');
    end
  end
  if (existfile(fullfile(pi,cudartdyn)))
    if (existfile(fullfile(pi,'..',libfolder,cudartlib)))
      cudartdll = 1;
      CUDA_ROOT = fullfile(pi,'..');
    end
  end
  if (existfile(fullfile(pi,cufftdyn)))
    if (existfile(fullfile(pi,'..',libfolder,cufftlib)))
      cufftdll = 1;
      CUDA_ROOT = fullfile(pi,'..');
    end
  end
  
end


% return
y.cublasdll = cublasdll;
y.cudartdll = cudartdll;
y.cufftdll  = cufftdll;
y.CUDA_ROOT = CUDA_ROOT;
end

function y = existfile(file)

% first implementation
%y = exist(file,'file');
if (isunix)
  try 
    a=ls(file);
    y=1;
  catch err
    y=0;
  end
else
  y = ~isempty(ls(file));
end

end

function y = splitPath(path)
if (ispc)
  c = ';';
end
if (isunix)
  c = ':';
end

y={};
[t,r] = strtok(path,c);
y{end+1} = t;
while ~isempty(r)
  [t,r] = strtok(r,c);
  y{end+1} = t;
end


end
