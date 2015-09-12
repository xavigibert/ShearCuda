function CompileMEX (file)

% load nvidia settings (CUDA_ROOT, CC_BIN)
nv = GetCUDAInfo;
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

include = '';

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
common   = '';

%% Build
file = [file ' ' common];
cmd = ['mex ' mexflags ' -outdir ' outdir ' ' file ' ' allinclude ' ' alllib ];
disp(cmd);
eval(cmd);

end

