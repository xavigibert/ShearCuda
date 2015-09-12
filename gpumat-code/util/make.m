function make(varargin)

% save current dir
curdir = pwd;

% finds all subdirectories
subs = dir('.');

disp(['*** Current directory ' pwd ]);
file = fullfile('.','makefile.m');
if (exist(file,'file'))
  objs = makefile(varargin{:});
  makeObjs(objs);
else
  disp('-> nothing to do ');
  objs = [];
end


% look for moduleinit.m file
for i=1:length(subs)
  sub = subs(i);
  if (sub.isdir)&&(strcmp(sub.name,'.')==0)&&(strcmp(sub.name,'..')==0)
    % enter a new directory if there is a makefile.m or more folders
    if (exist(fullfile(sub.name,'makefile.m'),'file'))||(countSubFolders(fullfile('.',sub.name))>0)
      cd(sub.name);
      
      make(varargin{:});
      cd(curdir);
    end
  end
end
cd(curdir);
% file = fullfile('.','makefile.m');
% if (exist(file,'file'))
%   objs = makefile(varargin{:});
% else
%   disp('-> nothing to do ');
%   objs = [];
% end



end


function y=countSubFolders(path)
  subs = dir(path);
  y = 0;
  for i=1:length(subs)
    sub = subs(i);
    if (sub.isdir)&&(strcmp(sub.name,'.')==0)&&(strcmp(sub.name,'..')==0)
      y = y +1;
    end
    
  end
end

function makeObjs(objs)
% recursively search for makefile.m
for i=1:length(objs)
  obj=objs{i};
  % cpp
  if (obj.cpp.do==1)
    makecpp(obj.cpp.infiles, obj.cpp.outdir, obj.cpp.include, obj.cpp.lib, obj.cpp.flags, obj.cpp.clean);
  end
  
  if (obj.lib.do==1)
    makelib(obj.lib.libname, obj.lib.indir, obj.lib.outdir, obj.lib.clean);
  end
  
  if (obj.ins.do==1)
    makeinstall(obj.ins.filter, obj.ins.inpath, obj.ins.outpath, obj.ins.skipfiles);
  end
  
  if (obj.cud.do==1)
    makecuda(obj.cud.base, obj.cud.include, obj.cud.clean);
  end
  
end


end