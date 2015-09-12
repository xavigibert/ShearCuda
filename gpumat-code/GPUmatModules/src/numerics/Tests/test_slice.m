function test_slice

global GPUtest

try
  type = GPUtest.type;
catch
  error('GPUtest not initialized. Please use GPUtestInit.');
end

%%
GPUtestLOG('***********************************************',0);
gpufun = GPUtest.gpufun;
cpufun = GPUtest.cpufun;
txtfun = GPUtest.txtfun;

% GPUtest.type
% 1 - real/complex
% 2 - real
% 3 - complex

type = GPUtest.type;
switch type
  case 1
    rangei = [0 sqrt(-1)];
    rangej = [0 sqrt(-1)];
  case 2
    rangei = 0;
    rangej = 0;
  case 3
    rangei = sqrt(-1);
    rangej = sqrt(-1);
end

testfun = GPUtest.testfun;

if (GPUtest.checkCompiler==1)
  A = zeros(5,GPUsingle); % dummy
  R = zeros(5,GPUsingle); % dummy
  a=1; %dummy
  b=2; %dummy
  c=3; %dummy
  d=4; %dummy
  
  % slice with 1 argument
  GPUcompileStart('comp_slice1','-f','-verbose0',A,a)
  R=slice(A,a);
  GPUcompileStop(R)
  
  % slice with 2 argument
  GPUcompileStart('comp_slice2','-f','-verbose0',A,a,b)
  R=slice(A,a,b);
  GPUcompileStop(R)
  
  % slice with 3 argument
  GPUcompileStart('comp_slice3','-f','-verbose0',A,a,b,c)
  R=slice(A,a,b,c);
  GPUcompileStop(R)
  
  % slice with 4 argument
  GPUcompileStart('comp_slice4','-f','-verbose0',A,a,b,c,d)
  R=slice(A,a,b,c,d);
  GPUcompileStop(R)
  
  
  
end

%% Test slice
for f=1:length(cpufun)
  
  %%
  % Test for expected errors
  % The following code should generate errors, we catch them
  A = feval(gpufun{f},rand(3,3));
  
  try
    slice(A,1,1,1);
    GPUtestLOG('Expected error',1);
  catch
  end
  
  
  try
    slice(A,{int32(0:9)},0);
    GPUtestLOG('Expected error',1);
  catch
  end
  
  try
    slice(A,3,3,3);
    A(-1:10,1);
    GPUtestLOG('Expected error',1);
  catch
  end
  
  try
    slice(A,{int32(0:99)});
    GPUtestLOG('Expected error',1);
  catch
  end
  
  for i=rangei
    switch i
      case 0
        complexity = 'real';
      case sqrt(-1)
        complexity = 'complex';
    end
    
    GPUtestLOG(sprintf('**** Testing slice (%s,%s)',txtfun{f}, complexity),0);
    
    %%
    A = feval(gpufun{f},rand(3,3,3)+i*rand(3,3,3));
    Ah = feval(cpufun{f},A);
    
    if (GPUtest.checkCompiler==1)
      B=comp_slice1(A,[1,1,END]);
    else
      B=slice(A,[1,1,END]);
    end
    Bh=Ah(1:end);
    compareCPUGPU(Bh,B);
  
    %%
    A = feval(gpufun{f},rand(3,3,3)+i*rand(3,3,3));
    Ah = feval(cpufun{f},A);
    
    if (GPUtest.checkCompiler==1)
      GPUcompileStart('comp_tmp','-f','-verbose0',A)
      R=slice(A,[1,1,END]);
      GPUcompileStop(R)
      
      B = comp_tmp(A);
    else
      B=slice(A,[1,1,END]);
    end
    Bh=Ah(1:end);
    compareCPUGPU(Bh,B);
  
  
  
    %%
    A = feval(gpufun{f},rand(3,3,3)+i*rand(3,3,3));
    Ah = feval(cpufun{f},A);
    
    if (GPUtest.checkCompiler==1)
      B=comp_slice1(A,[1,1,END-1]);
    else
      B=slice(A,[1,1,END-1]);
    end
    Bh=Ah(1:end-1);
    compareCPUGPU(Bh,B);
    
    %%
    A = feval(gpufun{f},rand(3,3,3)+i*rand(3,3,3));
    Ah = feval(cpufun{f},A);
    
    if (GPUtest.checkCompiler==1)
      GPUcompileStart('comp_tmp','-f','-verbose0',A)
      R=slice(A,[1,1,END-1]);
      GPUcompileStop(R)
      
      B = comp_tmp(A);
    else
      B=slice(A,[1,1,END-1]);
    end
    Bh=Ah(1:end-1);
    compareCPUGPU(Bh,B);
    
    
    %%
    A = feval(gpufun{f},rand(3,3,3)+i*rand(3,3,3));
    Ah = feval(cpufun{f},A);
    
    if (GPUtest.checkCompiler==1)
      B=comp_slice1(A,[1,1,END-2]);
    else
      B=slice(A,[1,1,END-2]);
    end
    Bh=Ah(1:end-2);
    compareCPUGPU(Bh,B);
    
    %%
    A = feval(gpufun{f},rand(3,3,3)+i*rand(3,3,3));
    Ah = feval(cpufun{f},A);
    
    if (GPUtest.checkCompiler==1)
      GPUcompileStart('comp_tmp','-f','-verbose0',A)
      R=slice(A,[1,1,END-2]);
      GPUcompileStop(R)
      
      B = comp_tmp(A);
    else
      B=slice(A,[1,1,END-2]);
    end
    Bh=Ah(1:end-2);
    compareCPUGPU(Bh,B);
    
     %%
    A = feval(gpufun{f},rand(3,3,3)+i*rand(3,3,3));
    Ah = feval(cpufun{f},A);
    
    if (GPUtest.checkCompiler==1)
      GPUcompileStart('comp_tmp','-f','-verbose0',A)
      R=slice(A,':',':',END-2);
      GPUcompileStop(R)
      
      B = comp_tmp(A);
    else
      B=slice(A,':',':',END-2);
    end
    Bh=Ah(:,:,end-2);
    compareCPUGPU(Bh,B);
    
    %%
    A = feval(gpufun{f},rand(3,3,3)+i*rand(3,3,3));
    Ah = feval(cpufun{f},A);
    
    if (GPUtest.checkCompiler==1)
      B=comp_slice3(A,END,END,[1,1,END]);
    else
      B=slice(A,END,END,[1,1,END]);
    end
    Bh=Ah(end,end,1:end);
    compareCPUGPU(Bh,B);
    
    %%
    A = feval(gpufun{f},rand(3,3,3)+i*rand(3,3,3));
    Ah = feval(cpufun{f},A);
    
    if (GPUtest.checkCompiler==1)
      GPUcompileStart('comp_tmp','-f','-verbose0',A)
      R=slice(A,END,END,[1,1,END]);
      GPUcompileStop(R)
      
      B = comp_tmp(A);
    else
      B=slice(A,END,END,[1,1,END]);
    end
    Bh=Ah(end,end,1:end);
    compareCPUGPU(Bh,B);
    
    %%
    A = feval(gpufun{f},rand(3,3,3)+i*rand(3,3,3));
    Ah = feval(cpufun{f},A);
    
    if (GPUtest.checkCompiler==1)
      B=comp_slice3(A,END-1,END-1,[1,1,END]);
    else
      B=slice(A,END-1,END-1,[1,1,END]);
    end
    Bh=Ah(end-1,end-1,1:end);
    compareCPUGPU(Bh,B);
    
    %%
    A = feval(gpufun{f},rand(3,3,3)+i*rand(3,3,3));
    Ah = feval(cpufun{f},A);
    
    if (GPUtest.checkCompiler==1)
      GPUcompileStart('comp_tmp','-f','-verbose0',A)
      R=slice(A,END-1,END-1,[1,1,END]);
      GPUcompileStop(R)
      
      B = comp_tmp(A);
    else
      B=slice(A,END-1,END-1,[1,1,END]);
    end
    Bh=Ah(end-1,end-1,1:end);
    compareCPUGPU(Bh,B);
    
    %%
    A = feval(gpufun{f},rand(3,3,3)+i*rand(3,3,3));
    Ah = feval(cpufun{f},A);
    
    if (GPUtest.checkCompiler==1)
      B=comp_slice1(A,[END-2,-1,1]);
    else
      B=slice(A,[END-2,-1,1]);
    end
    Bh=Ah(end-2:-1:1);
    compareCPUGPU(Bh,B);
    
    %%
    A = feval(gpufun{f},rand(3,3,3)+i*rand(3,3,3));
    Ah = feval(cpufun{f},A);
    
    if (GPUtest.checkCompiler==1)
      GPUcompileStart('comp_tmp','-f','-verbose0',A)
      R=slice(A,[END-2,-1,1]);
      GPUcompileStop(R)
      
      B = comp_tmp(A);
    else
      B=slice(A,[END-2,-1,1]);
    end
    Bh=Ah(end-2:-1:1);
    compareCPUGPU(Bh,B);
    
    
    %%
    A = feval(gpufun{f},rand(3,3,3)+i*rand(3,3,3));
    Ah = feval(cpufun{f},A);
    
    if (GPUtest.checkCompiler==1)
      B=comp_slice3(A,{[1 1 1]},':',':');
    else
      B=slice(A,{[1 1 1]},':',':');
    end
    Bh=Ah([1 1 1],:,:);
    compareCPUGPU(Bh,B);
    
    %%
    A = feval(gpufun{f},rand(3,3,3)+i*rand(3,3,3));
    Ah = feval(cpufun{f},A);
    if (GPUtest.checkCompiler==1)
      B=comp_slice3(A,{[1 2 3]},':',':');
    else
      B=slice(A,{[1 2 3]},':',':');
    end
    Bh=Ah([1 2 3],:,:);
    compareCPUGPU(Bh,B);
    
    %%
    A = feval(gpufun{f},rand(3,3,3)+i*rand(3,3,3));
    Ah = feval(cpufun{f},A);
    if (GPUtest.checkCompiler==1)
      B=comp_slice3(A,{[1 3 2]},':',':');
    else
      B=slice(A,{[1 3 2]},':',':');
    end
    Bh=Ah([1 3 2],:,:);
    compareCPUGPU(Bh,B);
    
    %%
    A = feval(gpufun{f},rand(3,3,3)+i*rand(3,3,3));
    Ah = feval(cpufun{f},A);
    
    if (GPUtest.checkCompiler==1)
      B=comp_slice3(A,':',{[1 2 3]},':');
    else
      B=slice(A,':',{[1 2 3]},':');
    end
    Bh=Ah(:,[1 2 3],:);
    compareCPUGPU(Bh,B);
    
    %%
    A = feval(gpufun{f},rand(3,3,3)+i*rand(3,3,3));
    Ah = feval(cpufun{f},A);
    
    if (GPUtest.checkCompiler==1)
      B=comp_slice1(A,{[1 2 3]});
    else
      B=slice(A,{[1 2 3]});
    end
    Bh=Ah([1 2 3]);
    compareCPUGPU(Bh,B);
    
    %%
    A = feval(gpufun{f},rand(3,3,3)+i*rand(3,3,3));
    Ah = feval(cpufun{f},A);
    
    for kk=1:10
      if (GPUtest.checkCompiler==1)
        B=comp_slice1(A,{[kk 1 kk]});
      else
        B=slice(A,{[kk 1 kk]});
      end
      Bh=Ah([kk 1 kk]);
      compareCPUGPU(Bh,B);
      
    end
    
    %%
    A = feval(gpufun{f},rand(3,3,3)+i*rand(3,3,3));
    Ah = feval(cpufun{f},A);
    
    if (GPUtest.checkCompiler==1)
      B=comp_slice3(A,':',{[1 1 1]},':');
    else
      B=slice(A,':',{[1 1 1]},':');
    end
    Bh=Ah(:,[1 1 1],:);
    compareCPUGPU(Bh,B);
    
    if (GPUtest.fastMode==0)
      
      %%
      A = feval(gpufun{f},rand(2,4,3,3)+i*rand(2,4,3,3));
      Ah = feval(cpufun{f},A);
      
      if (GPUtest.checkCompiler==1)
        B=comp_slice1(A,[60,3,70]);
      else
        B=slice(A,[60,3,70]);
      end
      Bh = Ah(60:3:70);
      compareCPUGPU(Bh,B);
      
      %%
      A = feval(gpufun{f},rand(2,4,3,3)+i*rand(2,4,3,3));
      Ah = feval(cpufun{f},A);
      
      if (GPUtest.checkCompiler==1)
        B=comp_slice1(A,[60,1,70]);
      else
        B=slice(A,[60,1,70]);
      end
      Bh = Ah(60:70);
      compareCPUGPU(Bh,B);
      
      %%
      A = feval(gpufun{f},rand(2100,2100)+i*rand(2100,2100));
      Ah = feval(cpufun{f},A);
      
      
      for kk=1:floor((numel(A)-300)/15):(numel(A)-300)
        offset = kk;
        Rh = Ah((1+offset):(256+offset));
        if (GPUtest.checkCompiler==1)
          R0 = comp_slice1(A,[(1+offset),1,(256+offset)]);
        else
          R0 = slice(A,[(1+offset),1,(256+offset)]);
        end
        compareCPUGPU(Rh,R0);
        
      end
      
      %%
      A = feval(gpufun{f},rand(4100,4100));
      Ah = feval(cpufun{f},A);
      
      
      for kk=1:floor((numel(A)-300)/10):(numel(A)-300)
        offset = kk;
        Rh = Ah((1+offset):(256+offset));
        if (GPUtest.checkCompiler==1)
          R0 = comp_slice1(A,[(1+offset),1,(256+offset)]);
        else
          R0 = slice(A,[(1+offset),1,(256+offset)]);
        end
        compareCPUGPU(Rh,R0);
        
      end
      
      %%
      A = feval(gpufun{f},rand(2100,2100)+i*rand(2100,2100));
      Ah = feval(cpufun{f},A);
      
      s = size(A);
      for kk=1:floor(s(2)/10):s(2)
        Rh = Ah(:,kk);
        if (GPUtest.checkCompiler==1)
          R0 = comp_slice2(A,':',kk);
        else
          R0 = slice(A,':',kk);
        end
        compareCPUGPU(Rh,R0);
        
      end
      
      %%
      A = feval(gpufun{f},rand(200,200,200)+i*rand(200,200,200));
      Ah = feval(cpufun{f},A);
      
      s = size(A);
      for kk=1:floor(s(3)/10):s(3)
        Rh = Ah(:,:,kk);
        if (GPUtest.checkCompiler==1)
          R0 = comp_slice3(A,':',':',kk);
        else
          R0 = slice(A,':',':',kk);
        end
        compareCPUGPU(Rh,R0);
        
      end
      
      %%
      A = feval(gpufun{f},rand(4100,4100));
      Ah = feval(cpufun{f},A);
      
      s = size(A);
      for kk=1:floor(s(1)/10):s(1)
        Rh = Ah(kk,:);
        if (GPUtest.checkCompiler==1)
          R0 = comp_slice2(A,kk,':');
        else
          R0 = slice(A,kk,':');
        end
        compareCPUGPU(Rh,R0);
        
      end
      
      %%
      A = feval(gpufun{f},rand(2100,2100)+i*rand(2100,2100));
      Ah = feval(cpufun{f},A);
      
      s = size(A);
      for kk=1:floor(s(1)/10):s(1)
        Rh = Ah(kk,:);
        if (GPUtest.checkCompiler==1)
          R0 = comp_slice2(A,kk,':');
        else
          R0 = slice(A,kk,':');
        end
        compareCPUGPU(Rh,R0);
        
      end
      
      %%
      if strcmp(txtfun{f},'single')
        A = ones(4100,4100,feval(gpufun{f}));
      else
        A = ones(3100,3100,feval(gpufun{f}));
      end
      Ah = feval(cpufun{f},A);
      if (GPUtest.checkCompiler==1)
        B=comp_slice1(A,[1,1,END]);
      else
        B=slice(A,[1,1,END]);
      end
      Bh = Ah(1:end);
      
      clear A
      clear Ah
      compareCPUGPU(Bh,B);
      
      %%
      A = feval(gpufun{f},rand(3,3,3)+i*rand(3,3,3));
      Ah = feval(cpufun{f},A);
      if (GPUtest.checkCompiler==1)
        B=comp_slice3(A,':',':',1);
      else
        B=slice(A,':',':',1);
      end
      Bh = Ah(:,:,1);
      compareCPUGPU(Bh,B);
      
      %%
      A = feval(gpufun{f},rand(2,4,3,3)+i*rand(2,4,3,3));
      Ah = feval(cpufun{f},A);
      if (GPUtest.checkCompiler==1)
        B=comp_slice4(A,[1,1,2],[1,1,2],[1,1,2],[1,1,2]);
      else
        B=slice(A,[1,1,2],[1,1,2],[1,1,2],[1,1,2]);
      end
      Bh = Ah(1:2,1:2,1:2,1:2);
      compareCPUGPU(Bh,B);
      
      %%
      if (GPUtest.checkCompiler==1)
        B=comp_slice2(A,[1,1,2],[1,1,END]);
      else
        B=slice(A,[1,1,2],[1,1,END]);
      end
      Bh = Ah(1:2,1:end);
      compareCPUGPU(Bh,B);
      
      %%
      if (GPUtest.checkCompiler==1)
        B=comp_slice1(A,[1,1,END]);
      else
        B=slice(A,[1,1,END]);
      end
      Bh = Ah(1:end);
      compareCPUGPU(Bh,B);
      
      %%
      if (GPUtest.checkCompiler==1)
        B=comp_slice3(A,1,[1,1,2],[1,1,END]);
      else
        B=slice(A,1,[1,1,2],[1,1,END]);
      end
      Bh = Ah(1,1:2,1:end);
      compareCPUGPU(Bh,B);
      
      %%
      if (GPUtest.checkCompiler==1)
        B=comp_slice2(A,1,[1,1,END]);
      else
        B=slice(A,1,[1,1,END]);
      end
      Bh = Ah(1,1:end);
      compareCPUGPU(Bh,B);
      
      %%
      if (GPUtest.checkCompiler==1)
        B=comp_slice4(A,1,[1,1,2],END,END);
      else
        B=slice(A,1,[1,1,2],END,END);
      end
      Bh = Ah(1,1:2,end,end);
      compareCPUGPU(Bh,B);
      
      %%
      A = feval(gpufun{f},rand(2000,2000));
      Ah = feval(cpufun{f},A);
      if (GPUtest.checkCompiler==1)
        B=comp_slice2(A,[1,1,END],[1,1,100]);
      else
        B=slice(A,[1,1,END],[1,1,100]);
      end
      Bh=Ah(1:end,1:100);
      compareCPUGPU(Bh,B);
      
      %%
      if (GPUtest.checkCompiler==1)
        B=comp_slice2(A,[1,1,END],[1,1,300]);
        
      else
        B=slice(A,[1,1,END],[1,1,300]);
      end
      Bh=Ah(1:end,1:300);
      compareCPUGPU(Bh,B);
      
      %%
      if (GPUtest.checkCompiler==1)
        B=comp_slice2(A,[1,1,END],[1,1,700]);
        
      else
        B=slice(A,[1,1,END],[1,1,700]);
      end
      Bh=Ah(1:end,1:700);
      compareCPUGPU(Bh,B);
      
      
      %%
      A = feval(gpufun{f},rand(5,4,3)+i*rand(5,4,3));
      Ah = feval(cpufun{f},A);
      if (GPUtest.checkCompiler==1)
        B=comp_slice3(A,[1,1,2],[1,1,2],[1,1,2]);
        
      else
        B=slice(A,[1,1,2],[1,1,2],[1,1,2]);
      end
      Bh = Ah(1:2,1:2,1:2);
      compareCPUGPU(Bh,B);
      
      
      %%
      A = feval(gpufun{f},rand(5,4,3)+i*rand(5,4,3));
      Ah = feval(cpufun{f},A);
      if (GPUtest.checkCompiler==1)
        B=comp_slice1(A,[1,1,END]);
        
      else
        B=slice(A,[1,1,END]);
      end
      Bh = Ah(1:end);
      compareCPUGPU(Bh,B);
      
      
      %%
      A = feval(gpufun{f},rand(1000,1000,3)+i*rand(1000,1000,3));
      Ah = feval(cpufun{f},A);
      if (GPUtest.checkCompiler==1)
        B=comp_slice2(A,[1,1,END],[1,1,100]);
        
      else
        B=slice(A,[1,1,END],[1,1,100]);
      end
      Bh=Ah(1:end,1:100);
      compareCPUGPU(Bh,B);
      
      %%
      if (GPUtest.checkCompiler==1)
        B=comp_slice3(A,[1,1,END],[1,1,100],[1,1,3]);
        
      else
        B=slice(A,[1,1,END],[1,1,100],[1,1,3]);
      end
      Bh=Ah(1:end,1:100,1:3);
      compareCPUGPU(Bh,B);
      
      %%
      if (GPUtest.checkCompiler==1)
        B=comp_slice3(A,[1,1,END],[1,1,100],END);
        
      else
        B=slice(A,[1,1,END],[1,1,100],END);
      end
      Bh=Ah(1:end,1:100,end);
      compareCPUGPU(Bh,B);
      
      
      %%
      if (GPUtest.checkCompiler==1)
        B=comp_slice3(A,[1,1,END],[1,1,100],':');
        
      else
        B=slice(A,[1,1,END],[1,1,100],':');
      end
      Bh=Ah(1:end,1:100,:);
      compareCPUGPU(Bh,B);
      
      %%
      if (GPUtest.checkCompiler==1)
        B=comp_slice3(A,':',[1,1,100],':');
        
      else
        B=slice(A,':',[1,1,100],':');
      end
      Bh=Ah(:,1:100,:);
      compareCPUGPU(Bh,B);
      
      %%
      if (GPUtest.checkCompiler==1)
        B=comp_slice3(A,':',':',':');
        
      else
        B=slice(A,':',':',':');
      end
      Bh=Ah(:,:,:);
      compareCPUGPU(Bh,B);
      
      
      %%
      A = feval(gpufun{f},rand(1000,1000,3)+i*rand(1000,1000,3));
      Ah = feval(cpufun{f},A);
      
      if (GPUtest.checkCompiler==1)
        B=comp_slice1(A,{[1 2; 3 4]});
        
      else
        B=slice(A,{[1 2; 3 4]});
      end
      Bh=Ah([1 2; 3 4]);
      compareCPUGPU(Bh,B);
      
      
      %%
      if (GPUtest.checkCompiler==1)
        B=comp_slice1(A,{[1:100;2:101;3:102]});
      else
        B=slice(A,{[1:100;2:101;3:102]});
      end
      Bh=Ah([1:100;2:101;3:102]);
      compareCPUGPU(Bh,B);
      
      
      
      %%
      A = feval(gpufun{f},rand(1000,1000,3)+i*rand(1000,1000,3));
      Ah = feval(cpufun{f},A);
      if (GPUtest.checkCompiler==1)
        B=comp_slice2(A,[900,-10,800],':');
      else
        B=slice(A,[900,-10,800],':');
      end
      Bh=Ah(900:-10:800,:);
      compareCPUGPU(Bh,B);
      
      
      %%
      A = feval(gpufun{f},rand(1000,1000,3)+i*rand(1000,1000,3));
      Ah = feval(cpufun{f},A);
      
      if (GPUtest.checkCompiler==1)
        B=comp_slice2(A,[900,-1,800],':');
      else
        B=slice(A,[900,-1,800],':');
      end
      Bh=Ah(900:-1:800,:);
      compareCPUGPU(Bh,B);
      
    end
    
    
    
    
  end
  
  %% Test particular cases
  u = GPUsingle(reshape(1:25,5,5));
  GPUcompileStart('comp_test','-f','-verbose0',u)
  UP=slice(u,[2,1,END-1],[1,1,END-2])+slice(u,[2,1,END-1],[3,1,END])+slice(u,[1,1,END-2],[2,1,END-1])+slice(u,[3,1,END],[2,1,END-1]);
  GPUcompileStop(UP)
  UP = comp_test(u);
  UP0 = slice(u,[2,1,END-1],[1,1,END-2])+slice(u,[2,1,END-1],[3,1,END])+slice(u,[1,1,END-2],[2,1,END-1])+slice(u,[3,1,END],[2,1,END-1]);
  compareCPUGPU(single(UP0),UP);
  
  
end
GPUtestLOG('***********************************************',0);
end
