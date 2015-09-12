function objs = makefile(varargin)

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
  
  lib = '';
  common   = '';
  include = ['-I"' fullfile('..','..','include') '" -I"' fullfile('..','cuda','latest','cudalib') '" -I"' cuda.incpath '" -I"."'];
  
  if (make_debug)
    flags = '-c -g -DDEBUG -DMATLAB ';
  else
    flags = '-c -DMATLAB ';
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
  indir = fullfile('.');
  outdir = fullfile('.');
  libname = 'cudalib';
  objs{end+1} = maketask('lib',libname,indir, outdir, make_clean);
end

%% make install
if (make_install)
  
  
end










end
