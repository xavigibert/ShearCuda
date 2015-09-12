function test_memCpyHtoD

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

% Currently complex is not supported and not used in this test
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
  a = 1; % dummy
  b = 2; % dummy
  c = 3; % dummy
  GPUcompileStart('comp_memCpyHtoD','-f','-verbose0',A,a,b,c)
  memCpyHtoD(A,a,b,c);
  GPUcompileStop
end

%% Test memCpyHtoD
for f=1:length(cpufun)
  
  
  
  for i=rangei
    switch i
      case 0
        complexity = 'real';
      case sqrt(-1)
        complexity = 'complex';
    end
    
    
    GPUtestLOG(sprintf('**** Testing memCpyHtoD (%s,%s)',txtfun{f}, complexity),0);
    
    %% Check errors
    A = feval(gpufun{f},rand(10,10));
    B = feval(cpufun{f},rand(10,10));
    Ah = feval(cpufun{f},A);
    Bh = feval(cpufun{f},B);
    
    try
      memCpyHtoD(A,B,101,100);
      error('Error expected');
    catch
      
    end
    
    try
      memCpyHtoD(A,B,1,101);
      error('Error expected');
    catch
      
    end
    
    try
      memCpyHtoD(A,B,0,101);
      error('Error expected');
    catch
      
    end
    %%
    
    A = feval(gpufun{f},rand(1000,1000));
    B = feval(cpufun{f},rand(1000,1000));
    Ah = feval(cpufun{f},A);
    Bh = feval(cpufun{f},B);
    
    for k=1:10000:100000
      if (GPUtest.checkCompiler==1)
        comp_memCpyHtoD(A,B,k,100);
      else
        memCpyHtoD(A,B,k,100);
      end
      Ah(k:k+99) = Bh(1:100);
      compareCPUGPU(Ah,A);
      
    end
    
    %%
    
    A = feval(gpufun{f},rand(1000,1000));
    B = feval(cpufun{f},rand(1000,1000));
    Ah = feval(cpufun{f},A);
    Bh = feval(cpufun{f},B);
    
    if (GPUtest.checkCompiler==1)
      comp_memCpyHtoD(A,B,1,1e6);
    else
      memCpyHtoD(A,B,1,1e6);
    end
    Ah(1:end) = Bh(1:end);
    compareCPUGPU(Ah,A);
    
    
    
  end
  
  
end
GPUtestLOG('***********************************************',0);
end
