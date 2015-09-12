function test_eye
% testeye eye test

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
  GPUtestLOG(sprintf('**** Testing eye (%s)  ',txtfun{f}),0);
  
  if (GPUtest.checkCompiler==1)
    A = GPUsingle(rand(5)); % dummy
    k = 1; % dummy
    z = 2; % dummy
    GPUcompileStart('comp_eye','-f','-verbose0',k, z, A)
    R = eye(k,z,A);
    GPUcompileStop(R)
  end
  
  for k=20:30:100
    Ah = eye(k,k,txtfun{f});
    if (GPUtest.checkCompiler==1)
      R = comp_eye(k,k,feval(gpufun{f}));
    else
      R = eye(k,k,feval(gpufun{f}));
    end
    compareCPUGPU(Ah,R);
    
    Ah = eye(k,40,txtfun{f});
    if (GPUtest.checkCompiler==1)
      R = comp_eye(k,40, feval(gpufun{f}));
    else
      R = eye(k,40,feval(gpufun{f}));
    end
    compareCPUGPU(Ah,R);
    
    Ah = eye(200,k,txtfun{f});
    if (GPUtest.checkCompiler==1)
      R = comp_eye(200,k, feval(gpufun{f}));
    else
      R = eye(200,k,feval(gpufun{f}));
    end
    compareCPUGPU(Ah,R);
  end
  
  
end
GPUtestLOG('***********************************************',0);
end
