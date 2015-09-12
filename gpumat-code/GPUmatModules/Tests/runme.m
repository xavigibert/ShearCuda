function runme

global GPUtest

try
  type = GPUtest.type;
catch
  error('GPUtest not initialized. Please use GPUtestInit.');
end

curdir = pwd;
%% Manual files
if (GPUtest.checkCompiler==1)
  GPUtestLOG('**** Modules - Manual files',0);
  GPUtestLOG('**** Warning: compilation mode not tested',0);
  GPUtestLOG('***********************************************',0);
  
else
  cd Manual
  make cpp
  runme
  cd(curdir)
end

%% Examples from the Function reference
if (GPUtest.checkCompiler==1)
  GPUtestLOG('**** Modules - Examples from function reference',0);
  GPUtestLOG('**** Warning: compilation mode not tested',0);
  GPUtestLOG('***********************************************',0);
  
else
  cd mhelp
  ls=dir('test*.m');
  for i=1:length(ls)
    n = ls(i).name(1:end-2);
    startMemoryCheck
    eval(n);
    stopMemoryCheck
  end
  cd(curdir)
end

%% release code
arch = computer('arch');

%% Examples
cd ..
cd ..
cd release
cd(arch)
cd modules
cd Examples

%% Examples/numerics
cd numerics
if (GPUtest.checkCompiler==1)
  GPUtestLOG('**** Modules - Examples/numerics',0);
  GPUtestLOG('**** Warning: compilation mode not tested',0);
  GPUtestLOG('***********************************************',0);
  
else
  ls=dir('test*.m');
  for i=1:length(ls)
    n = ls(i).name(1:end-2);
    startMemoryCheck
    eval(n);
    stopMemoryCheck
  end
  startMemoryCheck
  runme
  stopMemoryCheck
end
cd ..

%% Examples/GPUtype
cd GPUtype

if (GPUtest.checkCompiler==1)
  GPUtestLOG('**** Modules - Examples/GPUtype',0);
  GPUtestLOG('**** Warning: compilation mode not tested',0);
  GPUtestLOG('***********************************************',0);
  
else
  ls=dir('test*.m');
  for i=1:length(ls)
    n = ls(i).name(1:end-2);
    startMemoryCheck
    eval(n);
    stopMemoryCheck
  end
  startMemoryCheck
  runme
  stopMemoryCheck
end

cd ..

%% Examples/codeopt
cd codeopt
%addpath(pwd);
cd Tests

if (GPUtest.checkCompiler==1)
  GPUtestLOG('**** Modules - Examples/codeopt',0);
  GPUtestLOG('**** Warning: compilation mode not tested',0);
  GPUtestLOG('***********************************************',0);
  
else
  
  ls=dir('test*.m');
  for i=1:length(ls)
    n = ls(i).name(1:end-2);
    if (GPUtest.checkPointers)
      % do nothing
    else
      eval(n);
    end
  end
end

cd ..
%rmpath(pwd);
cd ..
cd ..

%% rand
cd rand
cd Tests
ls=dir('test*.m');
for i=1:length(ls)
  n = ls(i).name(1:end-2);
  
  if (GPUtest.memLeak==1)
    m = GPUmem;
  end
  
  startMemoryCheck
  eval(n);
  
  if (GPUtest.memLeak==1)
    mnew = GPUmem;
    if (mnew < m*0.9)
      error('Check memory');
    end
  end
  
  stopMemoryCheck
  
end
cd ..
cd ..

%% numerics
cd numerics
cd Tests
ls=dir('test*.m');
for i=1:length(ls)
  n = ls(i).name(1:end-2);
  
  if (GPUtest.memLeak==1)
    m = GPUmem;
  end
  
  startMemoryCheck
  eval(n);
  
  if (GPUtest.memLeak==1)
    mnew = GPUmem;
    if (mnew < m*0.9)
      error('Check memory');
    end
  end
  
  stopMemoryCheck
  
end
cd ..
cd Examples

startMemoryCheck
runme
stopMemoryCheck

cd(curdir)


