function makecpp(infiles, outdir, include, lib, flags, clean)

% default
if (~exist('clean','var'))
  clean = 0;
end

% some parameters
extmex = ['.' mexext];
if (isunix)
  extobj = '.o';
else
  extobj = '.obj';
end

force = 0;

% analyze flags
options = parseOptions(flags);
flags = '';
for i=1:length(options)
  option = options{i};
  switch option
    case '-force'
    case '-c'
      flags = [flags ' ' option];
      % no mex, -c option generates obj
      extmex = extobj;
    otherwise
      flags = [flags ' ' option];
  end
  
end

% check for outpath. Create if necessary
if (~exist(outdir,'dir'))
  disp(['mkdir -> '  outdir ]);
  mkdir(outdir);
end


% for some reason the include for Visual Studio is not
% included. This problem was reported but it cannot be reproduced
% all the times.

clinclude = '';
switch computer
  case {'PCWIN64'}
    
    clinclude = locateCL;
  case {'PCWIN'}
    
    clinclude = locateCL;
  case {'GLNXA64'}
    
  otherwise
    
end


%% Build
for i=1:length(infiles)
  
  % check if file needs to be compiled according to date
  % infiles{i} can be a single file or a sequence of files, for example:
  % numerics.cpp
  % numerics.cpp GPUmat.cpp
  
  files = parseFiles(infiles{i});
  compile = 0;
  
  for j=1:length(files)
    file = files{j};
    infile = java.io.File(file);
    dot = infile.toString.lastIndexOf('.');
    sep = infile.toString.lastIndexOf(java.io.File.separatorChar);
    if (dot~=-1)
      fileext  = char(infile.toString.substring(dot+1));
      filename = char(infile.toString.substring(sep+1,dot));
      % the output file is given by the first file in list
      if (j==1)
        outfile = java.io.File(fullfile(outdir, [filename extmex]));
        if (clean==1)
          filename = fullfile(outdir, [filename extmex]);
          if (exist(filename,'file'))
            cmd = ['delete ' filename ];
            disp(cmd);
            eval(cmd);
          end
        end
      end
      if (infile.lastModified > outfile.lastModified)
        compile = 1;
      end
    end
  end
  %   infile = java.io.File(infiles{i});
  %   dot = infile.toString.lastIndexOf('.');
  %   sep = infile.toString.lastIndexOf(java.io.File.separatorChar);
  %   fileext  = char(infile.toString.substring(dot+1));
  %   filename = char(infile.toString.substring(sep+1,dot));
  %
  %   outfile = java.io.File(fullfile(outdir, [filename extmex]));
  
  if (clean ==1)
  elseif (compile==1)
    cmd = ['mex ' flags ' -outdir ' outdir ' ' infiles{i} ' ' clinclude ' ' include ' ' lib ];
    %cmd = ['mex ' mexflags ' -outdir ' outdir ' ' infiles{i} ' ' clinclude ' ' allinclude ' ' alllib ];
    disp(cmd);
    eval(cmd);
  else
    disp([infiles{i} '-> nothing to be done']);
  end
  
end


end

function y = parseOptions(options)
c = ' ';

y={};
[t,r] = strtok(options,c);
y{end+1} = t;
while ~isempty(r)
  [t,r] = strtok(r,c);
  y{end+1} = t;
end


end

function y = parseFiles(files)
y = parseOptions(files);

end

