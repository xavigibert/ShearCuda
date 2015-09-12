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
include = ['-I"' fullfile('..','include') '"'];
include = [include ' -I"' cuda.incpath '"'];

%% flags
if (make_debug)
  flags = '-g ';
else
  flags = '';
end
if (isunix)
  flags = [flags ' -DUNIX'];
end
%% make .cpp files
if (make_cpp)
  
  lib = '';
  lib = [lib ' -L"' cuda.libpath '"'];
  lib = [lib ' -lcurand '];
  lib = [lib ' -lcublas -lcuda -lcudart -lcufft'];
  
  outdir = '.';
  
  common   = fullfile('..','common','GPUmat.cpp');
  infiles = {};
  infiles{end+1} = ['RANDModuleManager.cpp ' common ];
  
  % make
  objs{end+1} = maketask('cpp',infiles, outdir, include, lib, flags, make_clean);
  
  outdir = 'GPUtype';
  common   = fullfile('..','common','GPUmat.cpp');
  infiles = {};
  infiles{end+1} = [fullfile('GPUtype','rand.cpp') ' ' common ];
  infiles{end+1} = [fullfile('GPUtype','GPUrand.cpp') ' ' common ];
  infiles{end+1} = [fullfile('GPUtype','randn.cpp') ' '  common ];
  infiles{end+1} = [fullfile('GPUtype','GPUrandn.cpp') ' '  common ];
  
  objs{end+1} = maketask('cpp',infiles, outdir, include, lib, flags, make_clean);
  
end

%% make cuda kernels
if (make_cuda)
%   objs{end+1} = maketask('cud','texture',include, make_clean);
end


%% make lib
if (make_lib)
end

%% make install
if (make_install)
  skipfiles = {'makefile.m'};
  
  % INSTALL main folder
  inpath = '.';
  outpath = fullfile(compilepar.releasepath,'modules','rand');
  filesfilters = {['*.' mexext], '*.cubin', 'moduleinit.m'};
  objs{end+1} = maketask('ins',filesfilters, inpath, outpath, skipfiles);
   
  % INSTALL in @GPUtype folder
  inpath = 'GPUtype';
  outpath = fullfile(compilepar.releasepath,'@GPUtype');
  filesfilters = {['*.' mexext], '*.m'};
  objs{end+1} = maketask('ins',filesfilters, inpath, outpath, skipfiles);
   
  % INSTALL Tests
  inpath = 'Tests';
  outpath = fullfile(compilepar.releasepath,'modules','rand','Tests');
  filesfilters = {['*.' mexext], '*.m'};
  objs{end+1} = maketask('ins',filesfilters, inpath, outpath, skipfiles);
  
end








end
