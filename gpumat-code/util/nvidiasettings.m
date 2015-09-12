function cuda = nvidiasettings
% specify in CUDA_ROOT the directory where CUDA is installed
% specify in CUDA_SDK  the directory where CUDA SDK is installed

str = computer('arch');
cuda.sdkpath  = getenv('CUDA_SDK_PATH');
cuda.path     = getenv('CUDA_PATH');

% if (isempty(cuda.sdkpath))
%   error('CUDA_SDK_PATH system variable is not properly configured');
% end

if (isempty(cuda.path))
  error('CUDA_PATH system variable is not properly configured');
end

CUDA_ROOT = cuda.path;

% include and library path
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

cuda.libpath = fullfile(CUDA_ROOT, libfolder);
cuda.incpath = fullfile(CUDA_ROOT, 'include');

% SUPPORTED CUDA ARCHITECTURE
cuda.arch = {'10','11','12','13','20','21','30','35'};

% check folders
if (~exist(CUDA_ROOT,'dir'))
  error('The specified CUDA_ROOT folder is invalid');
  
end

% check for lib and include
if (~exist([CUDA_ROOT filesep 'lib'],'dir'))
  error('The specified CUDA_ROOT folder is invalid');
end

if (~exist([CUDA_ROOT filesep 'include'],'dir'))
  error('The specified CUDA_ROOT folder is invalid');
end


%disp('NVIDIA settings OK');

end
