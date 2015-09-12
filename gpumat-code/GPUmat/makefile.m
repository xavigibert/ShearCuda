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

if (make_clean==0)
  % absolute path of release path
  arch = computer('arch');
  
  %inspath = fullfile('..','release',arch);
  inspath = fullfile('..','..','GPUmat');
  % check for outpath. Create if necessary
  if (~exist(inspath,'dir'))
    disp(['mkdir -> '  inspath ]);
    mkdir(inspath);
  end
  
  
  compilepar.releasepath = getAbsolutePath(inspath);
  
end

%% make .cpp files
if (make_cpp)
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
  
  inpath = fullfile('src','release');
  outpath = fullfile(compilepar.releasepath);
  filesfilters = {''};
  objs{end+1} = maketask('ins', filesfilters, inpath, outpath, skipfiles);
  
  arch = computer('arch');
  inpath = fullfile('VisualRedist',arch,'etc');
  outpath = fullfile(compilepar.releasepath,'etc');
  filesfilters = {''};
  objs{end+1} = maketask('ins', filesfilters, inpath, outpath, skipfiles);

  
end



end

