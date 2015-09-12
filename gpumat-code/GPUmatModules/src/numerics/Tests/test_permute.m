function test_permute

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
  a=1; %dummy
  
  % 
  GPUcompileStart('comp_permute','-f','-verbose0',A,a)
  R=permute(A,a);
  GPUcompileStop(R)
  
  
end

%% Test repmat
for f=1:length(cpufun)
  
  
  
  for i=rangei
    switch i
      case 0
        complexity = 'real';
      case sqrt(-1)
        complexity = 'complex';
    end
    
    GPUtestLOG(sprintf('**** Testing permute (%s,%s)',txtfun{f}, complexity),0);
    
    %%
    A = feval(gpufun{f},rand(1000,1000)+i*rand(1000,1000));
    Ah = feval(cpufun{f},A);
    p = [2 1];
    if (GPUtest.checkCompiler==1)
      R = comp_permute(A,p);
    else
      R = permute(A,p);
    end
    
    Rh = permute(Ah,p);
    compareCPUGPU(Rh,R);
    
    %%
    A = feval(gpufun{f},rand(1000,1000)+i*rand(1000,1000));
    Ah = feval(cpufun{f},A);
    p = [1 2];
    if (GPUtest.checkCompiler==1)
      R = comp_permute(A,p);
    else
      R = permute(A,p);
    end
    
    Rh = permute(Ah,p);
    compareCPUGPU(Rh,R);
    
    %%
    A = feval(gpufun{f},rand(100,100,10)+i*rand(100,100,10));
    Ah = feval(cpufun{f},A);
    p = [2 1 3];
    if (GPUtest.checkCompiler==1)
      R = comp_permute(A,p);
    else
      R = permute(A,p);
    end
    
    Rh = permute(Ah,p);
    compareCPUGPU(Rh,R);
    
    %%
    A = feval(gpufun{f},rand(100,100,10)+i*rand(100,100,10));
    Ah = feval(cpufun{f},A);
    p = [1 2 3];
    if (GPUtest.checkCompiler==1)
      R = comp_permute(A,p);
    else
      R = permute(A,p);
    end
    
    Rh = permute(Ah,p);
    compareCPUGPU(Rh,R);
    
    %%
    A = feval(gpufun{f},rand(100,100,10)+i*rand(100,100,10));
    Ah = feval(cpufun{f},A);
    p = [3 2 1];
    if (GPUtest.checkCompiler==1)
      R = comp_permute(A,p);
    else
      R = permute(A,p);
    end
    
    Rh = permute(Ah,p);
    compareCPUGPU(Rh,R);
    
    %%
    A = feval(gpufun{f},rand(100,100,10)+i*rand(100,100,10));
    Ah = feval(cpufun{f},A);
    p = [3 1 2];
    if (GPUtest.checkCompiler==1)
      R = comp_permute(A,p);
    else
      R = permute(A,p);
    end
    
    Rh = permute(Ah,p);
    compareCPUGPU(Rh,R);
    
    %%
    A = feval(gpufun{f},rand(100,10,10,20)+i*rand(100,10,10,20));
    Ah = feval(cpufun{f},A);
    p = [3 1 2 4];
    if (GPUtest.checkCompiler==1)
      R = comp_permute(A,p);
    else
      R = permute(A,p);
    end
    
    Rh = permute(Ah,p);
    compareCPUGPU(Rh,R);
    
    %%
    A = feval(gpufun{f},rand(100,10,10,20)+i*rand(100,10,10,20));
    Ah = feval(cpufun{f},A);
    p = [1 3 2 4];
    if (GPUtest.checkCompiler==1)
      R = comp_permute(A,p);
    else
      R = permute(A,p);
    end
    
    Rh = permute(Ah,p);
    compareCPUGPU(Rh,R);
    
    %%
    A = feval(gpufun{f},rand(100,10,10,20)+i*rand(100,10,10,20));
    Ah = feval(cpufun{f},A);
    p = [3 1 4 2];
    if (GPUtest.checkCompiler==1)
      R = comp_permute(A,p);
    else
      R = permute(A,p);
    end
    
    Rh = permute(Ah,p);
    compareCPUGPU(Rh,R);
    
    %%
    A = feval(gpufun{f},rand(100,10,10,20)+i*rand(100,10,10,20));
    Ah = feval(cpufun{f},A);
    p = [3 4 2 1];
    if (GPUtest.checkCompiler==1)
      R = comp_permute(A,p);
    else
      R = permute(A,p);
    end
    
    Rh = permute(Ah,p);
    compareCPUGPU(Rh,R);
    
    %%
    A = feval(gpufun{f},rand(100,10,10,20)+i*rand(100,10,10,20));
    Ah = feval(cpufun{f},A);
    p = [2 1 3 4];
    if (GPUtest.checkCompiler==1)
      R = comp_permute(A,p);
    else
      R = permute(A,p);
    end
    
    Rh = permute(Ah,p);
    compareCPUGPU(Rh,R);
    
    
    
    
    
    
    
  end
  
  
  
  
end
GPUtestLOG('***********************************************',0);
end
