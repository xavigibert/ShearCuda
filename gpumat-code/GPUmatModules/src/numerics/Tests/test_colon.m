function test_colon
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
type = GPUtest.type;
if (type == 1)
  rangei = [0 1];
end


if (GPUtest.checkCompiler==1)
  A = zeros(5,GPUsingle); % dummy
  R = zeros(5,GPUsingle); % dummy
  a=1; %dummy
  b=2; %dummy
  c=3; %dummy
  
  % repmat with 2 arguments
  GPUcompileStart('comp_colon2','-f','-verbose0',a,b,A)
  R=colon(a,b,A);
  GPUcompileStop(R)
  
  % repmat with 3 arguments
  GPUcompileStart('comp_colon3','-f','-verbose0',a,b,c,A)
  R=colon(a,b,c,A);
  GPUcompileStop(R)
  
end

for t=rangei
  if (t==0)
    type1= 'real';
  else
    type1 = 'complex';
  end
  for f=1:length(cpufun)
    GPUtestLOG(sprintf('**** Testing COLON (%s,%s) ',txtfun{f},type1),0);
    
    RR = feval(gpufun{f});
    
    if (GPUtest.bigKernel)
      % should work with big arrays
      J = 1;
      D = 1;
      K = 4100*4100;
      
      if (t==0)
        Ah = feval(cpufun{f},colon(J,D,K));
        if (GPUtest.checkCompiler==1)
          A = comp_colon3(J,D,K,RR);
        else
          A = colon(J,D,K,RR);
        end
        
      else
        Ah = complex(feval(cpufun{f},colon(J,D,K)));
        if (GPUtest.checkCompiler==1)
          A = comp_colon3(J,D,K,complex(RR));
        else
          A = colon(J,D,K,complex(RR));
        end
        
      end
      
      
      compareCPUGPU(Ah,A);
    end
    
    %%%%%%%%%%%%%%%%%%%
    J = 1e-4;
    K = 10e-4;
    D = 1e-4;
    
    if (t==0)
      Ah = feval(cpufun{f},colon(J,D,K));
      if (GPUtest.checkCompiler==1)
        A = comp_colon3(J,D,K,RR);
      else
        A = colon(J,D,K,RR);
      end
    else
      Ah = complex(feval(cpufun{f},colon(J,D,K)));
      if (GPUtest.checkCompiler==1)
        A = comp_colon3(J,D,K,complex(RR));
      else
        A = colon(J,D,K,complex(RR));
      end
      
    end
    
    
    compareCPUGPU(Ah,A);
    
    %%%%%%%%%%%%%%%%%%%
    J = 0.0;
    D = 0.1;
    
    for J=0.0:0.1:0.5
    for K=0.6:0.11:10.0
      Ah = feval(cpufun{f},colon(J,D,K));
      A = colon(J,D,K,RR);
      compareCPUGPU(Ah,A);
      
    end
    end
    
    %%%%%%%%%%%%%%%%%%%
    J = 0.0;
    D = 0.1234;
    
    for J=0.0:0.125:0.5
    for K=0.6:0.115:10.0
      Ah = feval(cpufun{f},colon(J,D,K));
      A = colon(J,D,K,RR);
      compareCPUGPU(Ah,A);
      
    end
    end
    
    %%%%%%%%%%%%%%%%%%%
    J = 0.0;
    D = -0.1234;
    
    for K=0.0:0.125:0.5
    for J=0.6:0.115:10.0
      Ah = feval(cpufun{f},colon(J,D,K));
      A = colon(J,D,K,RR);
      compareCPUGPU(Ah,A);
      
    end
    end
    
    %%%%%%%%%%%%%%%%%%%
    J = 1e-4;
    K = 10e-4;
    D = 1e-4;
    
    Ah = feval(cpufun{f},colon(J,D,K));
    if (GPUtest.checkCompiler==1)
      A = comp_colon3(J,D,K,RR);
    else
      A = colon(J,D,K,RR);
    end
    
    
    compareCPUGPU(Ah,A);
    
    
    %%%%%%%%%%%%%%%%%%%
    J = 1000;
    K = 1;
    D = 10;
    
    Ah = feval(cpufun{f},colon(J,D,K));
    if (GPUtest.checkCompiler==1)
      A = comp_colon3(J,D,K,RR);
    else
      A = colon(J,D,K,RR);
    end
    
    
    compareCPUGPU(Ah,A);
    
    %%%%%%%%%%%%%%%%%%%
    J = 1000;
    K = 1;
    D = -10;
    
    
    Ah = feval(cpufun{f},colon(J,D,K));
    if (GPUtest.checkCompiler==1)
      A = comp_colon3(J,D,K,RR);
    else
      A = colon(J,D,K,RR);
    end
    
    
    compareCPUGPU(Ah,A);
    
    %%%%%%%%%%%%%%%%%%%
    J = 1;
    K = 1000;
    D = 10;
    
    
    Ah = feval(cpufun{f},colon(J,D,K));
    if (GPUtest.checkCompiler==1)
      A = comp_colon3(J,D,K,RR);
    else
      A = colon(J,D,K,RR);
    end
    
    
    compareCPUGPU(Ah,A);
    
    %%%%%%%%%%%%%%%%%%%
    J = 1000;
    K = 1;
    
    
    Ah = feval(cpufun{f},colon(J,D,K));
    if (GPUtest.checkCompiler==1)
      A = comp_colon3(J,D,K,RR);
    else
      A = colon(J,D,K,RR);
    end
    
    
    compareCPUGPU(Ah,A);
    
    %%%%%%%%%%%%%%%%%%%
    J = 1;
    K = 100;
    
    
    Ah = feval(cpufun{f},colon(J,D,K));
    if (GPUtest.checkCompiler==1)
      A = comp_colon3(J,D,K,RR);
    else
      A = colon(J,D,K,RR);
    end
    
    
    compareCPUGPU(Ah,A);
    
    %%%%%%%%%%%%%%%%%%%
    J = [1000 10];
    K = [1 10];
    D = 10+sqrt(-1)*1;
    
    
    Ah = feval(cpufun{f},colon(J,D,K));
    if (GPUtest.checkCompiler==1)
      A = comp_colon3(J,D,K,RR);
    else
      A = colon(J,D,K,RR);
    end
    
    
    compareCPUGPU(Ah,A);
    
    %%%%%%%%%%%%%%%%%%%
    J = [1 10];
    K = [10 10];
    D = 1+sqrt(-1)*1;
    
    
    Ah = feval(cpufun{f},colon(J,D,K));
    if (GPUtest.checkCompiler==1)
      A = comp_colon3(J,D,K,RR);
    else
      A = colon(J,D,K,RR);
    end
    
    
    compareCPUGPU(Ah,A);
    
    %%%%%%%%%%%%%%%%%%%
    J = 1000;
    K = 1;
    
    
    Ah = feval(cpufun{f},colon(J,K));
    if (GPUtest.checkCompiler==1)
      A = comp_colon2(J,K,RR);
    else
      A = colon(J,K,RR);
    end
    
    
    compareCPUGPU(Ah,A);
    
    %%%%%%%%%%%%%%%%%%%
    J = 1000;
    K = 1;
    
    
    Ah = feval(cpufun{f},colon(J,K));
    if (GPUtest.checkCompiler==1)
      A = comp_colon2(J,K,RR);
    else
      A = colon(J,K,RR);
    end
    
    
    compareCPUGPU(Ah,A);
    
    %%%%%%%%%%%%%%%%%%%
    J = 1;
    K = 1000;
    
    
    Ah = feval(cpufun{f},colon(J,K));
    if (GPUtest.checkCompiler==1)
      A = comp_colon2(J,K,RR);
    else
      A = colon(J,K,RR);
    end
    
    
    compareCPUGPU(Ah,A);
    
    %%%%%%%%%%%%%%%%%%%
    J = 1000;
    K = 1;
    
    
    Ah = feval(cpufun{f},colon(J,K));
    if (GPUtest.checkCompiler==1)
      A = comp_colon2(J,K,RR);
    else
      A = colon(J,K,RR);
    end
    
    
    compareCPUGPU(Ah,A);
    
    %%%%%%%%%%%%%%%%%%%
    J = 1;
    K = 100;
    
    
    Ah = feval(cpufun{f},colon(J,K));
    if (GPUtest.checkCompiler==1)
      A = comp_colon2(J,K,RR);
    else
      A = colon(J,K,RR);
    end
    
    
    compareCPUGPU(Ah,A);
    
    %%%%%%%%%%%%%%%%%%%
    J = [1000 10];
    K = [1 10];
    
    
    Ah = feval(cpufun{f},colon(J,K));
    if (GPUtest.checkCompiler==1)
      A = comp_colon2(J,K,RR);
    else
      A = colon(J,K,RR);
    end
    
    
    compareCPUGPU(Ah,A);
    
    %%%%%%%%%%%%%%%%%%%
    J = [1 10];
    K = [10 10];
    
    
    Ah = feval(cpufun{f},colon(J,K));
    if (GPUtest.checkCompiler==1)
      A = comp_colon2(J,K,RR);
    else
      A = colon(J,K,RR);
    end
    
    
    compareCPUGPU(Ah,A);
    
  end
end
GPUtestLOG('***********************************************',0);
end
