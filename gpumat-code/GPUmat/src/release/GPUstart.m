function y = GPUstart(query)
%GPUstart Start GPU environment and load required components

%
%     Copyright (C) 2012  GP-you Group (http://gp-you.org)
% 
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <http://www.gnu.org/licenses/>.

% next line required for 2009a
warning off MATLAB:structOnObject


if (nargin==0)

else
  y = mislocked('GPUmat');
  return;
end

if (mislocked('GPUmat'))
  disp('GPU already started');
  return
end

type about.txt

disp('Starting GPU');

mypath = mfilename('fullpath');
mypath = mypath(1:end-length(mfilename)-1);

%add required folders to path
pathinc = { ...
  '', ...
  'matcublas', ...
  'matcuda', ...
  'matcufft', ...
  'matcudrv', ...
  'bin' ...
  'util' ...
  % fullfile('modules','util') ...
  %   'matcufft' ...
  };

for i=1:length(pathinc)
  addpath([mypath filesep pathinc{i}]);
end

disp('')
GPUmatVer = GPUmatVersion;
disp(['- GPUmat version: ' GPUmatVer.version]);
disp(['- Required CUDA version: ' GPUmatVer.cuda]);
disp('')

% Check MEX
try
testmex;
catch
  if (ispc)
    disp('Unable to run TEST MEX file. Microsoft Visual C++ 2008 Redistributable Package should be installed in your system.');
  end
  if (isunix)
    disp('Unable to run TEST MEX file.');
  end
  
  error('Please check the GPUmat User Guide for further help, or send an email to gp-you@gp-you.org.');   
end


% Path to cubin
try
GPUdrvInit(0);
catch
  disp('Unable to initialize CUDA driver. Running system diagnostics.');
  GPUmatSystemCheck;
  error('Unable to initialize CUDA driver. Please report this problem to gp-you@gp-you.org.');   
end

% Check for devices
devcount = 1;
try
  [status,devcount]=cudaGetDeviceCount(10);
catch
  disp('Unable to get GPU information. Running system diagnostics.');
  GPUmatSystemCheck;
  error('Unable to initialize CUDA driver. Please report this problem to gp-you@gp-you.org.');   
end

GPUinfo;
dev = 0;
if (devcount>1)
  disp('  - Your system has multiple GPUs installed'); 
  dev = input(['    -> Please specify the GPU device number to use [0-' num2str(devcount-1) ']: ']); 
  if (dev>=devcount)
    error('Specified device number is incorrect');
  end  
end

% Start GPUmanager
try
[status,major,minor] = cudaGetDeviceMajorMinor(dev);
catch
  disp('Unable to get GPU information. Running system diagnostics.');
  GPUmatSystemCheck;
  error('Unable to initialize CUDA driver. Please report this problem to gp-you@gp-you.org.');   
end

cudaCheckStatus(status,['Unable to get the compute capability for device N. ' num2str(dev) ]);

if (major==1)&&(minor==0)
  cubinpath = [mypath filesep 'cuda' filesep 'cudalib10'];
elseif (major==1)&&(minor==1)
  cubinpath = [mypath filesep 'cuda' filesep 'cudalib11'];
elseif (major==1)&&(minor==2)
  cubinpath = [mypath filesep 'cuda' filesep 'cudalib12'];
elseif (major==1)&&(minor==3)
  cubinpath = [mypath filesep 'cuda' filesep 'cudalib13'];
elseif (major==2)&&(minor==0)
  cubinpath = [mypath filesep 'cuda' filesep 'cudalib20'];
elseif (major==2)&&(minor==1)
  cubinpath = [mypath filesep 'cuda' filesep 'cudalib21'];
elseif (major==2)&&(minor==2)
  cubinpath = [mypath filesep 'cuda' filesep 'cudalib22'];
elseif (major==2)&&(minor==3)
  cubinpath = [mypath filesep 'cuda' filesep 'cudalib23'];
elseif (major==3)&&(minor==0)
  cubinpath = [mypath filesep 'cuda' filesep 'cudalib30'];
elseif (major==3)&&(minor==5)
  cubinpath = [mypath filesep 'cuda' filesep 'cudalib35'];
else
  % default
  if (major==1)
    cubinpath = [mypath filesep 'cuda' filesep 'cudalib10'];
  elseif (major==2)
    cubinpath = [mypath filesep 'cuda' filesep 'cudalib20'];
  elseif (major==3)
    cubinpath = [mypath filesep 'cuda' filesep 'cudalib30'];
  end  
end

cubinpath = [cubinpath '.cubin'];
try
  GPUmanager(0, cubinpath, dev, major, minor);
  disp(['  - CUDA compute capability ' num2str(major) '.' num2str(minor)']); 
catch
  err = lasterror;
  disp(err.message);
  
  disp(['Unable to load the kernels in file ' cubinpath '. Running system diagnostics.']);
  GPUmatSystemCheck;
  error(['Unable to load the kernels in file ' cubinpath '. Please report this problem to gp-you@gp-you.org.']);
end

% call GPUmat, just a dummy function
GPUmat;

ava_mem = 0;
tot_mem = 0;
[status,ava_mem, tot_mem]=cuMemGetInfo(ava_mem, tot_mem);
cuCheckStatus( status, 'Unable to retrieve device information.');

disp('...done');

% Load modules
GPUmatLoadModules(fullfile(GPUmatPath,'modules'));

end


function loadLibrary(so,h,name)
disp(['  - loading library ' name ]);

% load library
if (libisloaded(name))
  unloadlibrary(name);
end

loadlibrary(so,h);
disp('    ...done');


end

function unloadLibrary(name)

% load library
if (libisloaded(name))
  unloadlibrary(name);
end

end

function unloadalllib()

unloadLibrary('cufflib');
unloadLibrary('cudalib');
unloadLibrary('culib');
unloadLibrary('cublaslib');

end
