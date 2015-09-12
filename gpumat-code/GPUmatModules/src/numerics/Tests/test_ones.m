function test_ones

global GPUtest

try
  type = GPUtest.type;
catch
  error('GPUtest not initialized. Please use GPUtestInit.');
end
%%
GPUtestLOG('***********************************************',0);

if (GPUtest.checkCompiler==1)
  A = zeros(5,GPUsingle); % dummy
  R = zeros(5,GPUsingle); % dummy
  a=1; %dummy
  b=2; %dummy
  c=3; %dummy
  d=4; %dummy
  e=5; %dummy
  
  
  % ones with 1 argument
  GPUcompileStart('comp_ones1','-f','-verbose0',a,A)
  R=ones(a,A);
  GPUcompileStop(R)
  
  % ones with 2 argument
  GPUcompileStart('comp_ones2','-f','-verbose0',a,b,A)
  R=ones(a,b,A);
  GPUcompileStop(R)
  
  % ones with 3 argument
  GPUcompileStart('comp_ones3','-f','-verbose0',a,b,c,A)
  R=ones(a,b,c,A);
  GPUcompileStop(R)
  
  % ones with 4 argument
  GPUcompileStart('comp_ones4','-f','-verbose0',a,b,c,d,A)
  R=ones(a,b,c,d,A);
  GPUcompileStop(R)
  
  % ones with 5 argument
  GPUcompileStart('comp_ones5','-f','-verbose0',a,b,c,d,e,A)
  R=ones(a,b,c,d,e,A);
  GPUcompileStop(R)
  
  
  
end

%% testones
gpufun = GPUtest.gpufun;
cpufun = GPUtest.cpufun;
txtfun = GPUtest.txtfun;
%type = GPUtest.type;
type = 2;
rangei=0;
if (type == 1)
  rangei = [0 1];
end
for t=rangei
  if (t==0)
    type1= 'real';
  else
    type1 = 'complex';
  end
  for f=1:length(cpufun)
    GPUtestLOG(sprintf('**** Testing ONES (%s,%s)  ',txtfun{f},type1),0);
    if (t==0)
      Ah = ones(100,100,txtfun{f});
      if (GPUtest.checkCompiler==1)
        A = comp_ones2(100,100,feval(gpufun{f}));
      else
        A = ones(100,100,feval(gpufun{f}));
      end
    else
      Ah = complex(ones(100,100,txtfun{f}));
      if (GPUtest.checkCompiler==1)
        A = comp_ones2(100,100,complex(feval(gpufun{f})));
      else
        A = ones(100,100,complex(feval(gpufun{f})));
      end
      
    end
    
    
    compareCPUGPU(Ah+1,A+1);
    
    if (t==0)
      Ah = ones(500,100,txtfun{f});
      if (GPUtest.checkCompiler==1)
        A = comp_ones2(500,100,feval(gpufun{f}));
      else
        A = ones(500,100,feval(gpufun{f}));
      end
    else
      Ah = complex(ones(500,100,txtfun{f}));
      if (GPUtest.checkCompiler==1)
        A = comp_ones2(500,100,complex(feval(gpufun{f})));
      else
        A = ones(500,100,complex(feval(gpufun{f})));
      end
      
    end
    
    compareCPUGPU(Ah+1,A+1);
    
    if (t==0)
      Ah = ones(500,200,100,txtfun{f});
      if (GPUtest.checkCompiler==1)
        A = comp_ones3(500,200,100,feval(gpufun{f}));
      else
        A = ones(500,200,100,feval(gpufun{f}));
      end
    else
      Ah = complex(ones(500,200,100,txtfun{f}));
      if (GPUtest.checkCompiler==1)
        A = comp_ones3(500,200,100,complex(feval(gpufun{f})));
      else
        A = ones(500,200,100,complex(feval(gpufun{f})));
      end
      
    end
    
    compareCPUGPU(Ah+1,A+1);
    
    if (t==0)
      Ah = ones(10,10,2,3,txtfun{f});
      if (GPUtest.checkCompiler==1)
        A = comp_ones4(10,10,2,3,feval(gpufun{f}));
      else
        A = ones(10,10,2,3,feval(gpufun{f}));
      end
    else
      Ah = complex(ones(10,10,2,3,txtfun{f}));
      if (GPUtest.checkCompiler==1)
        A = comp_ones4(10,10,2,3,complex(feval(gpufun{f})));
      else
        A = ones(10,10,2,3,complex(feval(gpufun{f})));
      end
      
    end
    
    compareCPUGPU(Ah+1,A+1);
    
    if (t==0)
      Ah = ones(10,10,2,3,4,txtfun{f});
      if (GPUtest.checkCompiler==1)
        A = comp_ones5(10,10,2,3,4,feval(gpufun{f}));
      else
        A = ones(10,10,2,3,4,feval(gpufun{f}));
      end
    else
      Ah = complex(ones(10,10,2,3,4,txtfun{f}));
      if (GPUtest.checkCompiler==1)
        A = comp_ones5(10,10,2,3,4,complex(feval(gpufun{f})));
      else
        A = ones(10,10,2,3,4,complex(feval(gpufun{f})));
      end
      
    end
    
    compareCPUGPU(Ah+1,A+1);
    
    if (t==0)
      Ah = ones([500,100],txtfun{f});
      if (GPUtest.checkCompiler==1)
        A = comp_ones1([500,100],feval(gpufun{f}));
      else
        A = ones([500,100],feval(gpufun{f}));
      end
    else
      Ah = complex(ones([500,100],txtfun{f}));
      if (GPUtest.checkCompiler==1)
        A = comp_ones1([500,100],complex(feval(gpufun{f})));
      else
        A = ones([500,100],complex(feval(gpufun{f})));
      end
      
    end
    
    compareCPUGPU(Ah+1,A+1);
    
    if (t==0)
      Ah = ones([10,10,2,3,4,5,6],txtfun{f});
      A = ones([10,10,2,3,4,5,6],feval(gpufun{f}));
    else
      Ah = complex(ones([10,10,2,3,4,5,6],txtfun{f}));
      A = ones([10,10,2,3,4,5,6],complex(feval(gpufun{f})));
    end
    
    compareCPUGPU(Ah+1,A+1);
    
  end
end
GPUtestLOG('***********************************************',0);
end
