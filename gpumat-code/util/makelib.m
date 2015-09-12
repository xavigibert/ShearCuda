function makelib(libname, indir, outdir, clean)

% default
if (~exist('clean','var'))
  clean = 0;
end

if (isunix)
  libname = ['lib' libname '.a'];
else
  libname = [libname '.lib'];
end


%%disp('**** MAKE LIB ****');
if (clean==1)
  filename = fullfile(outdir, libname);
  if (exist(filename,'file'))
    cmd = ['delete ' filename ];
    disp(cmd);
    eval(cmd);
  end
else
  
  if (isunix)
    cmd = ['ar rcs ' fullfile(outdir, libname) ' ' fullfile(indir, '*.o')];
  else
    cmd = ['lib /OUT:' fullfile(outdir, libname) ' ' fullfile(indir, '*.obj')];
  end
  disp(cmd);
  system(cmd);
end

end
