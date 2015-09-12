function test_subsref
global GPUtest

try
  type = GPUtest.type;
catch
  error('GPUtest not initialized. Please use GPUtestInit.');
end
%%
GPUtestLOG('***********************************************',0);

if (GPUtest.checkCompiler==1)
  GPUtestLOG('**** Compilation mode not supported',1);
end

type = GPUtest.type;

if (type == 1)
  rangei = [0 sqrt(-1)];
end

if (type == 2)
  rangei = 0;
end

gpufun = GPUtest.gpufun;
cpufun = GPUtest.cpufun;

txtfun = GPUtest.txtfun;

for f=1:length(cpufun)
  
  %%
  % Test error
  A = feval(gpufun{f},rand(3,3));
  
  try
    A(1).type;
    GPUtestLOG('Expected error',1);
  catch
  end
  
  try
    A(1,1,2);
    GPUtestLOG('Expected error',1);
  catch
  end
  
  try
    idx = feval(gpufun{f},1+sqrt(-1)*2);
    A(idx);
    GPUtestLOG('Expected error',1);
  catch
  end
  
  try
    A(1:10,1);
    GPUtestLOG('Expected error',1);
  catch
  end
  
  try
    A(-1:10,1);
    GPUtestLOG('Expected error',1);
  catch
  end
  
  try
    A(1:100);
    GPUtestLOG('Expected error',1);
  catch
  end
  
  for i=rangei
    if (i==0)
      type1 = 'real';
    end
    if (i==sqrt(-1))
      type1 = 'complex';
    end
    GPUtestLOG(sprintf('**** Testing SUBSREF (%s,%s) ',txtfun{f},type1),0);
    switch txtfun{f}
      case 'single'
        NMAXF = 4100;
      case 'double'
        NMAXF = 4100/2;
      otherwise
        NMAXF = 4100;
    end
    
    
    %%
    A = feval(gpufun{f},rand(3,3,3)+i*rand(3,3,3));
    Ah = feval(cpufun{f},A);
    
    B=A(:);
    Bh=Ah(:);
    compareCPUGPU(Bh,B);
    
    
    %%
    A = feval(gpufun{f},rand(3,3,3)+i*rand(3,3,3));
    Ah = feval(cpufun{f},A);
    
    B=A([1 1 1],:,:);
    Bh=Ah([1 1 1],:,:);
    compareCPUGPU(Bh,B);
    
    
    %%
    A = feval(gpufun{f},rand(3,3,3)+i*rand(3,3,3));
    Ah = feval(cpufun{f},A);
    
    B=A([1 2 3],:,:);
    Bh=Ah([1 2 3],:,:);
    compareCPUGPU(Bh,B);
    
    
    %%
    A = feval(gpufun{f},rand(3,3,3)+i*rand(3,3,3));
    Ah = feval(cpufun{f},A);
    
    B=A([1 3 2],:,:);
    Bh=Ah([1 3 2],:,:);
    compareCPUGPU(Bh,B);
    
    
    %%
    A = feval(gpufun{f},rand(3,3,3)+i*rand(3,3,3));
    Ah = feval(cpufun{f},A);
    
    B=A(:,[1 2 3],:);
    Bh=Ah(:,[1 2 3],:);
    compareCPUGPU(Bh,B);
    
    
    %%
    A = feval(gpufun{f},rand(3,3,3)+i*rand(3,3,3));
    Ah = feval(cpufun{f},A);
    
    B=A([1 2 3]);
    Bh=Ah([1 2 3]);
    compareCPUGPU(Bh,B);
    
    
    %%
    A = feval(gpufun{f},rand(3,3,3)+i*rand(3,3,3));
    Ah = feval(cpufun{f},A);
    
    for kk=1:10
      B=A([kk 1 kk]);
      Bh=Ah([kk 1 kk]);
      compareCPUGPU(Bh,B);
      
    end
    %%
    A = feval(gpufun{f},rand(3,3,3)+i*rand(3,3,3));
    Ah = feval(cpufun{f},A);
    
    B=A(:,[1 1 1],:);
    Bh=Ah(:,[1 1 1],:);
    compareCPUGPU(Bh,B);
    
    if (GPUtest.fastMode==0)
      
      %%
      A = feval(gpufun{f},rand(2,4,3,3)+i*rand(2,4,3,3));
      Ah = feval(cpufun{f},A);
      
      B=A(60:3:70);
      Bh = Ah(60:3:70);
      compareCPUGPU(Bh,B);
      
      
      %%
      A = feval(gpufun{f},rand(2,4,3,3)+i*rand(2,4,3,3));
      Ah = feval(cpufun{f},A);
      
      B=A(60:70);
      Bh = Ah(60:70);
      compareCPUGPU(Bh,B);
      
      %%
      A = feval(gpufun{f},rand(2100,2100)+i*rand(2100,2100));
      Ah = feval(cpufun{f},A);
      
      
      for kk=1:floor((numel(A)-300)/15):(numel(A)-300)
        offset = kk;
        Rh = Ah((1+offset):(256+offset));
        R0 = A((1+offset):(256+offset));
        compareCPUGPU(Rh,R0);
        
      end
      
      %%
      A = feval(gpufun{f},rand(NMAXF,NMAXF));
      Ah = feval(cpufun{f},A);
      
      
      for kk=1:floor((numel(A)-300)/10):(numel(A)-300)
        offset = kk;
        Rh = Ah((1+offset):(256+offset));
        R0 = A((1+offset):(256+offset));
        compareCPUGPU(Rh,R0);
        
      end
      
      %%
      A = feval(gpufun{f},rand(2100,2100)+i*rand(2100,2100));
      Ah = feval(cpufun{f},A);
      
      s = size(A);
      for kk=1:floor(s(2)/10):s(2)
        Rh = Ah(:,kk);
        R0 = A(:,kk);
        compareCPUGPU(Rh,R0);
        
      end
      
      %%
      A = feval(gpufun{f},rand(200,200,200)+i*rand(200,200,200));
      Ah = feval(cpufun{f},A);
      
      s = size(A);
      for kk=1:floor(s(3)/10):s(3)
        Rh = Ah(:,:,kk);
        R0 = A(:,:,kk);
        compareCPUGPU(Rh,R0);
        
      end
      
      %%
      A = feval(gpufun{f},rand(NMAXF,NMAXF));
      Ah = feval(cpufun{f},A);
      
      s = size(A);
      for kk=1:floor(s(1)/10):s(1)
        Rh = Ah(kk,:);
        R0 = A(kk,:);
        compareCPUGPU(Rh,R0);
        
      end
      
      %%
      A = feval(gpufun{f},rand(2100,2100)+i*rand(2100,2100));
      Ah = feval(cpufun{f},A);
      
      s = size(A);
      for kk=1:floor(s(1)/10):s(1)
        Rh = Ah(kk,:);
        R0 = A(kk,:);
        compareCPUGPU(Rh,R0);
        
      end
      
      %% compare more than 65000*256 threads
      %A = feval(gpufun{f},rand(4100,4100)+0*rand(4100,4100));
      A = ones(NMAXF,NMAXF,feval(gpufun{f}));
      Ah = feval(cpufun{f},A);
      
      B=A(1:end);
      Bh = Ah(1:end);
      
      clear A
      clear Ah
      compareCPUGPU(Bh,B);
      
      
      %%
      % Testing conditions similar to:
      % A = GPUsingle(rand(10,10,10));
      % A(1,1,1,1)
      A = feval(gpufun{f},rand(3,3,3)+i*rand(3,3,3));
      Ah = feval(cpufun{f},A);
      
      B=A(1,:,1,1,:); %% range is bigger than dimensions
      Bh = Ah(1,:,1,1,:);
      compareCPUGPU(Bh,B);
      
      %%
      A = feval(gpufun{f},rand(3,3,3)+i*rand(3,3,3));
      Ah = feval(cpufun{f},A);
      
      B=A(:,:,1);
      Bh = Ah(:,:,1);
      compareCPUGPU(Bh,B);
      
      
      %%
      A = feval(gpufun{f},rand(2,4,3,3)+i*rand(2,4,3,3));
      Ah = feval(cpufun{f},A);
      
      B=A(1:2,1:2,1:2,1:2);
      Bh = Ah(1:2,1:2,1:2,1:2);
      compareCPUGPU(Bh,B);
      
      
      B=A(1:2,1:end);
      Bh = Ah(1:2,1:end);
      compareCPUGPU(Bh,B);
      
      
      B=A(1:end);
      Bh = Ah(1:end);
      compareCPUGPU(Bh,B);
      
      
      B=A(1,1:2,1:end);
      Bh = Ah(1,1:2,1:end);
      compareCPUGPU(Bh,B);
      
      
      B=A(1,1:end);
      Bh = Ah(1,1:end);
      compareCPUGPU(Bh,B);
      
      
      B=A(1,end,end,end);
      Bh = Ah(1,end,end,end);
      compareCPUGPU(Bh,B);
      
      
      A = feval(gpufun{f},rand(2000,2000));
      Ah = feval(cpufun{f},A);
      
      B=A(1:end,1:100);
      Bh=Ah(1:end,1:100);
      compareCPUGPU(Bh,B);
      
      
      B=A(1:end,1:300);
      Bh=Ah(1:end,1:300);
      compareCPUGPU(Bh,B);
      
      
      B=A(1:end,1:700);
      Bh=Ah(1:end,1:700);
      compareCPUGPU(Bh,B);
      
      
      %%
      A = feval(gpufun{f},rand(5,4,3)+i*rand(5,4,3));
      Ah = feval(cpufun{f},A);
      
      B=A(1:2,1:2,1:2);
      Bh = Ah(1:2,1:2,1:2);
      compareCPUGPU(Bh,B);
      
      
      %%
      A = feval(gpufun{f},rand(5,4,3)+i*rand(5,4,3));
      Ah = feval(cpufun{f},A);
      
      B=A(1:end);
      Bh = Ah(1:end);
      compareCPUGPU(Bh,B);
      
      
      %%
      A = feval(gpufun{f},rand(1000,1000,3)+i*rand(1000,1000,3));
      Ah = feval(cpufun{f},A);
      
      B=A(1:end,1:100);
      Bh=Ah(1:end,1:100);
      compareCPUGPU(Bh,B);
      
      
      B=A(1:end,1:100,1:3);
      Bh=Ah(1:end,1:100,1:3);
      compareCPUGPU(Bh,B);
      
      
      B=A(1:end,1:100,end);
      Bh=Ah(1:end,1:100,end);
      compareCPUGPU(Bh,B);
      
      
      
      B=A(1:end,1:100,:);
      Bh=Ah(1:end,1:100,:);
      compareCPUGPU(Bh,B);
      
      
      B=A(:,1:100,:);
      Bh=Ah(:,1:100,:);
      compareCPUGPU(Bh,B);
      
      
      B=A(:,:,:);
      Bh=Ah(:,:,:);
      compareCPUGPU(Bh,B);
      
      
      %%
      A = feval(gpufun{f},rand(1000,1000,3)+i*rand(1000,1000,3));
      Ah = feval(cpufun{f},A);
      
      
      B=A([1 2; 3 4]);
      Bh=Ah([1 2; 3 4]);
      compareCPUGPU(Bh,B);
      
      
      B=A([1:100;2:101;3:102]);
      Bh=Ah([1:100;2:101;3:102]);
      compareCPUGPU(Bh,B);
      
      
      B=A();
      Bh=Ah();
      compareCPUGPU(Bh,B);
      
      
      %%
      idx = feval(gpufun{f},[1 2; 3 4]);
      B=A(idx);
      Bh=Ah([1 2; 3 4]);
      compareCPUGPU(Bh,B);
      
      
      idx = feval(gpufun{f},[1:100;2:101;3:102]);
      B=A(idx);
      Bh=Ah([1:100;2:101;3:102]);
      compareCPUGPU(Bh,B);
      
      
      
      %%
      A = feval(gpufun{f},rand(2,4,3,3)+i*rand(2,4,3,3));
      Ah = feval(cpufun{f},A);
      
      B=A(feval(gpufun{f},1:2),1:2,feval(gpufun{f},1:2),1:2);
      Bh = Ah(1:2,1:2,1:2,1:2);
      compareCPUGPU(Bh,B);
      
      
      B=A(feval(gpufun{f},1:2),1:end);
      Bh = Ah(1:2,1:end);
      compareCPUGPU(Bh,B);
      
      
      B=A(1:end);
      Bh = Ah(1:end);
      compareCPUGPU(Bh,B);
      
      
      B=A(1,feval(gpufun{f},1:2),1:end);
      Bh = Ah(1,1:2,1:end);
      compareCPUGPU(Bh,B);
      
      
      B=A(1,1:end);
      Bh = Ah(1,1:end);
      compareCPUGPU(Bh,B);
      
      
      B=A(1,end,end,end);
      Bh = Ah(1,end,end,end);
      compareCPUGPU(Bh,B);
      
      
      %%
      A = feval(gpufun{f},rand(2000,2000)+i*rand(2000,2000));
      Ah = feval(cpufun{f},A);
      
      B=A(1:end,feval(gpufun{f},1:100));
      Bh=Ah(1:end,1:100);
      compareCPUGPU(Bh,B);
      
      
      B=A(1:end,feval(gpufun{f},1:300));
      Bh=Ah(1:end,1:300);
      compareCPUGPU(Bh,B);
      
      
      B=A(1:end,1:700);
      Bh=Ah(1:end,1:700);
      compareCPUGPU(Bh,B);
      
      
      %%
      A = feval(gpufun{f},rand(1000,1000)+i*rand(1000,1000));
      Ah = feval(cpufun{f},A);
      
      idx = feval(gpufun{f},1:10000);
      B=A(idx);
      Bh = Ah(feval(cpufun{f},idx));
      compareCPUGPU(Bh,B);
      
      
      %%
      A = feval(gpufun{f},rand(1000,1000)+i*rand(1000,1000));
      Ah = feval(cpufun{f},A);
      
      idx = feval(gpufun{f},1:100);
      B=A(idx,idx);
      Bh = Ah(feval(cpufun{f},idx),feval(cpufun{f},idx));
      compareCPUGPU(Bh,B);
      
      
      %%
      A = feval(gpufun{f},rand(100,100,100)+i*rand(100,100,100));
      Ah = feval(cpufun{f},A);
      
      idx = feval(gpufun{f},1:100);
      B=A(idx,idx);
      Bh = Ah(feval(cpufun{f},idx),feval(cpufun{f},idx));
      compareCPUGPU(Bh,B);
      
      
      %%
      A = feval(gpufun{f},rand(3,3,3)+i*rand(3,3,3));
      Ah = feval(cpufun{f},A);
      
      idx = feval(gpufun{f},1:3);
      B=A(idx,idx);
      Bh = Ah(feval(cpufun{f},idx),feval(cpufun{f},idx));
      compareCPUGPU(Bh,B);
      
      
      %%
      A = feval(gpufun{f},rand(3,3,3)+i*rand(3,3,3));
      Ah = feval(cpufun{f},A);
      
      idx = feval(gpufun{f},1:3);
      B=A(idx,1:end);
      Bh = Ah(feval(cpufun{f},idx),1:end);
      compareCPUGPU(Bh,B);
      
      
      %%
      A = feval(gpufun{f},rand(1000,1000,3)+i*rand(1000,1000,3));
      Ah = feval(cpufun{f},A);
      
      
      B=A(900:-10:800,:);
      Bh=Ah(900:-10:800,:);
      compareCPUGPU(Bh,B);
      
      
      %%
      A = feval(gpufun{f},rand(1000,1000,3)+i*rand(1000,1000,3));
      Ah = feval(cpufun{f},A);
      
      
      B=A(900:-1:800,:);
      Bh=Ah(900:-1:800,:);
      compareCPUGPU(Bh,B);
    end
  end
  GPUtestLOG('***********************************************',0);
end