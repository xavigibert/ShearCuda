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


%% make .cpp files
if (make_cpp)
  arch = computer('arch');
  lib = '';
  lib = [lib ' -L"' cuda.libpath '"'];
  lib = [lib ' -L"' fullfile('..','lib') '"'];
  lib = [lib ' -L"' fullfile('..','cuda','latest','cudalib') '"'];
  lib = [lib ' -lcublas -lcuda -lcudart  -lgputype -lcudalib  -lcufft'];
  
  common   = '';
  include = ['-I"' cuda.incpath '" -I"."'];
  include = [include ' -I"' fullfile('..','..','include') '"'];
  include = [include ' -I"' fullfile('..','cuda','latest','cudalib') '"'];
  
  
  if (make_debug)
    flags = '-g -DDEBUG -DMATLAB ';
  else
    flags = '-DMATLAB ';
  end
  
  if (isunix)
    flags = [flags ' -DUNIX'];
  end
  
  outdir = fullfile('.');
  infiles = {};
  d = dir(fullfile('.','*.cpp'));
  for i=1:length(d)
    infiles{end+1} = [fullfile('.',d(i).name) ' ' common];
  end
  % make
  objs{end+1} = maketask('cpp',infiles, outdir, include, lib, flags, make_clean);
  
  
end

%% make cuda kernels
if (make_cuda)
end


%% make lib
if (make_lib)
end

%% make install
if (make_install)
  skipfiles = {'makefile.m'};
  
  inpath = fullfile('.');
  outpath = fullfile(compilepar.releasepath,'bin');
  filesfilters = {['*.' mexext], '*.m'};
  objs{end+1} = maketask('ins', filesfilters, inpath, outpath, skipfiles);
end








end
