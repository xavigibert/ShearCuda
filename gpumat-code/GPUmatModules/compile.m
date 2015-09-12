function compile


global compilepar
% absolute path of release path
arch = computer('arch');

inspath = fullfile('..','..','GPUmat');
% check for outpath. Create if necessary
if (~exist(inspath,'dir'))
  disp(['mkdir -> '  inspath ]);
  mkdir(inspath);
end


compilepar.releasepath = getAbsolutePath(inspath);


%make cpp cuda clean
make cpp cuda install


end


