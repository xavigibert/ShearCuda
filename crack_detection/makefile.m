function objs = makefile(varargin)

global compilepar

make_cpp = 0;
make_cuda = 0;
make_install = 0;
make_debug = 0;
make_lib = 0;
make_clean = 0;

cuda = nvidiasettings;

% objs is a list of tasks to be executed
objs ={};

for i=1:length(varargin)
  target = varargin{i};
  switch target
    case 'all'
      make_cpp = 1;
      make_cuda = 1;
      make_install = 1;
      make_lib = 1;
      
    case 'cpp'
      make_cpp = 1;
    case 'cuda'
      make_cuda = 1;
    case 'install'
      make_install = 1;
    case 'debug'
      make_debug = 1;
    case 'lib'
      make_lib = 1;
    case 'clean'
      make_clean = 1;
      
    otherwise
      error('Wrong option');
  end
end

arch = computer('arch');
include = ['-I"' fullfile('..','GPUmat','modules','include') '"'];
include = [include ' -I"' cuda.incpath '" -I"' fullfile('..','shearcuda','src_cuda') '"'];

%% flags
if (make_debug)
  flags = '-g ';
  cuda_flags = '-g -G ';
else
  flags = '';
  cuda_flags = '';
end
if (isunix)
  flags = [flags ' -DUNIX'];
end
%% make .cpp files
if (make_cpp)
  
  lib = '';
  lib = [lib ' -L"' cuda.libpath '"'];
  lib = [lib ' -lcublas -lcuda -lcudart -lcufft -lnpp'];
  
  outdir = '.';
  
  common   = fullfile('..','GPUmat','modules','common','GPUmat.cpp');
  common   = ['src_cuda/CracksGpuTimes.cpp src_cuda/CracksCudaFunctions.cpp ' ...
      '../shearcuda/src_cuda/GpuTimes.cpp ../shearcuda/src_cuda/ShearCudaFunctions.cpp ' common];
  
  infiles = {};
  infiles{end+1} = ['src_cuda/llCrackFeatures.cpp ' common ];
  infiles{end+1} = ['src_cuda/cracks_timers.cpp ' common ];
  infiles{end+1} = ['src_cuda/gc_crack_affinities.cpp ' common ];
  infiles{end+1} = ['src_cuda/cracks_gc.cpp ' common ];
  infiles{end+1} = ['src/ecm.cpp'];

  % make
  objs{end+1} = maketask('cpp',infiles, outdir, include, lib, flags, make_clean);
  
  
  
  
end

%% make cuda kernels
if (make_cuda)
  objs{end+1} = maketask('cud','cracks_cuda',include, make_clean, cuda_flags);
end


%% make lib
if (make_lib)
end

%% make install
if (make_install)
  skipfiles = {};
  
  inpath = '.';
  outpath = fullfile(compilepar.releasepath,'shearcuda');
  
  filesfilters = {['*.' mexext], 'README', '*.cpp','*.cu', '*.cubin', '*.m', 'moduleinit.m'};
  makeinstall(filesfilters, inpath, outpath)
  objs{end+1} = maketask('ins',filesfilters, inpath, outpath, skipfiles);
  
  
end








end
