function GPUmatSystemCheck()
%GPUSYSTEMCHECK Check environment
%   Perform the following checks on the system:
%   0) Print system information
%   1) CUDA settings
%   2) GPUmat settings


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
% Check CUDA settings

% System architecture
archstr = computer('arch');

gmat = GPUmatVersion;
fprintf('*** GPUmat system diagnostics\n');
fprintf('* Running on           -> "%s"\n', archstr);
fprintf('* Matlab ver.          -> "%s"\n', version);
fprintf('* GPUmat version       -> %s\n',gmat.version);
fprintf('* GPUmat build         -> %s\n',gmat.builddate);
fprintf('* GPUmat architecture  -> "%s"\n',gmat.arch);



% Check consistency in architecture
disp(' ');
fprintf('*** ARCHITECTURE TEST\n');
if (~strcmp(archstr,gmat.arch))
  fprintf('*** WARNING: GPUmat and SYSTEM architecture are different.\n');
else
  fprintf('*** GPUmat architecture test -> passed.\n');
end

disp(' ');
try
  fprintf('*** CUDA TEST\n');
  % Check CUDA
  cublasdll = 0;
  cudartdll = 0;
  cufftdll  = 0;
  cublasdllpath = 0;
  cudartdllpath = 0;
  cufftdllpath  = 0;
  
  
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
  end
  
  if (isunix)
    cublasdyn = 'libcublas.so';
    cudartdyn = 'libcudart.so';
    cufftdyn  = 'libcufft.so';
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
      cublasdll = 1;
      cublasdllpath = pi;
    end
    if (existfile(fullfile(pi,cudartdyn)))
      cudartdll = 1;
      cudartdllpath = pi;
    end
    if (existfile(fullfile(pi,cufftdyn)))
      cufftdll = 1;
      cufftdllpath = pi;
    end
  end
  
  if (cublasdll==0)
    fprintf('*** CUDA CUBLAS -> not installed.\n');
  else
    fprintf('*** CUDA CUBLAS -> installed (%s).\n', fullfile(cublasdllpath,cublasdyn));
  end
  
  if (cufftdll==0)
    fprintf('*** CUDA CUFFT  -> not installed.\n');
  else
    fprintf('*** CUDA CUFFT  -> installed (%s).\n', fullfile(cufftdllpath,cufftdyn));
  end
  
  if (cudartdll==0)
    fprintf('*** CUDA CUDART -> not installed.\n');
  else
    fprintf('*** CUDA CUDART -> installed (%s).\n', fullfile(cudartdllpath,cudartdyn));
  end
  
  
catch
  fprintf('*** GPUmatSystemCheck INTERNAL ERROR. PLEASE REPORT TO gp-you@gp-you.org.\n');
end

% Check device
fprintf('\n*** GPUmat device check\n');
GPUfullInfo;


end
function y = existfile(file)

% first implementation
%y = exist(file,'file');

y = ~isempty(ls(file));
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
