function task = maketask(varargin)

task.cpp.do = 0;
task.lib.do = 0;
task.ins.do = 0;
task.cud.do = 0;

type=varargin{1};
switch type
  case 'cud'
    task.cud.do = 1;
    task.cud.base = varargin{2};
    task.cud.include  = varargin{3};
    task.cud.clean    = varargin{4};
    
  case 'cpp'
    task.cpp.do = 1;
    task.cpp.infiles = varargin{2};
    task.cpp.outdir  = varargin{3};
    task.cpp.include = varargin{4};
    task.cpp.lib     = varargin{5};
    task.cpp.flags   = varargin{6};
    task.cpp.clean   = varargin{7};
    
  case 'ins'
    task.ins.do = 1;
    task.ins.filter = varargin{2};
    task.ins.inpath = varargin{3};
    task.ins.outpath = varargin{4};
    task.ins.skipfiles = varargin{5};
    
  case 'lib'
    task.lib.do = 1;
    task.lib.libname = varargin{2};
    task.lib.indir  = varargin{3};
    task.lib.outdir = varargin{4};
    task.lib.clean = varargin{5};
end


end