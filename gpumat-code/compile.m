function compile

%% Initial check



global compilepar
% absolute path of release path
arch = computer('arch');

% check for compiler
if (ispc)
res = system('cl');
if (res~=0)
  error('Unable to find Visual Studio compiler cl.exe. Please check your path. From a cmd shell type "cl" to check if the compiler is installed.');
end

% res = system('lib');
% if (res~=0)
%   error('Unable to find Visual Studio library manager lib.exe. Please check your path. From a cmd shell type "lib" to check if the library manager is installed.');
% end
end

%inspath = fullfile('.','release',arch);
inspath = fullfile('..','GPUmat');
% check for outpath. Create if necessary
if (~exist(inspath,'dir'))
  disp(['mkdir -> '  inspath ]);
  mkdir(inspath);
end


compilepar.releasepath = getAbsolutePath(inspath);

% Compilation sequence is important
curdir = pwd;
cd('GPUmat')
compile
cd(curdir)

cd('GPUmatModules')
compile
cd(curdir)

cd('matCUDA')
compile
cd(curdir)

cd('util')
make install
cd(curdir)

cd('doc')
make install
cd(curdir)

% Generate version number
GPUmatVer = '0.280';
CUDAver = '5.0';
makeVersion(fullfile(compilepar.releasepath, 'bin', 'GPUmatVersion.m'),GPUmatVer, CUDAver);

% Copy license
src = fullfile('.' , 'license.txt');
dst = fullfile(compilepar.releasepath);

if (ispc)
  copyfile(src,dst);
else
  system(['cp ' src ' ' dst]);
end

% create a ZIP/TAR file

src = compilepar.releasepath;

if (ispc)
  dst = ['GPUmat_' GPUmatVer '_' arch '_build' datestr(now,'yymmdd') '_CUDA_' CUDAver '.zip'];
  zip(dst,src)
end
if (isunix)
  dst = ['GPUmat_' GPUmatVer '_' arch '_build' datestr(now,'yymmdd') '_CUDA_' CUDAver '.tgz'];
  tar(dst,src)
end



end


function makeVersion(filename,ver, cudaver)

fid = fopen(filename,'w+');
fprintf(fid,'function y=GPUmatVersion()\n');
fprintf(fid,'y.version=''%s'';\n',ver);
fprintf(fid,'y.builddate=''%s'';\n',date);
fprintf(fid,'y.arch=''%s'';\n', computer('arch'));
fprintf(fid,'y.cuda=''%s'';\n',cudaver);

fclose(fid);
end


