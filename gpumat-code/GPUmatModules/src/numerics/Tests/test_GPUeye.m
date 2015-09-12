function test_GPUeye
% testGPUeye GPUeye test

global GPUtest

try
  type = GPUtest.type;
catch
  error('GPUtest not initialized. Please use GPUtestInit.');
end

%% testGPUeye
GPUtestLOG('***********************************************',0);
gpufun = GPUtest.gpufun;
cpufun = GPUtest.cpufun;
txtfun = GPUtest.txtfun;

for f=1:length(cpufun)
  GPUtestLOG(sprintf('**** Testing GPUeye (%s)  ',txtfun{f}),0);
  
  if (GPUtest.checkCompiler==1)
    R = zeros(5,GPUsingle); % dummy
    GPUcompileStart('comp_GPUeye','-f','-verbose0',R)
    GPUeye(R);
    GPUcompileStop
  end
  
  for k=20:30:100
    Ah = eye(k,k,txtfun{f});
    R = zeros(size(Ah),feval(gpufun{f}));
    if (GPUtest.checkCompiler==1)
      comp_GPUeye(R);
    else
      GPUeye(R);
    end
    compareCPUGPU(Ah,R);
    
    Ah = eye(k,40,txtfun{f});
    R = zeros(size(Ah),feval(gpufun{f}));
    if (GPUtest.checkCompiler==1)
      comp_GPUeye(R);
    else
      GPUeye(R);
    end
    compareCPUGPU(Ah,R);
    
    Ah = eye(200,k,txtfun{f});
    R = zeros(size(Ah),feval(gpufun{f}));
    if (GPUtest.checkCompiler==1)
      comp_GPUeye(R);
    else
      GPUeye(R);
    end
    compareCPUGPU(Ah,R);
  end
  
  
end
GPUtestLOG('***********************************************',0);
end
