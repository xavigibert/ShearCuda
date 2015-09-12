function test_zeros

global GPUtest

try
  type = GPUtest.type;
catch
  error('GPUtest not initialized. Please use GPUtestInit.');
end
%%
GPUtestLOG('***********************************************',0);

%% testzeros
gpufun = GPUtest.gpufun;
cpufun = GPUtest.cpufun;
txtfun = GPUtest.txtfun;
type = GPUtest.type;
type = 2;
rangei = 0;
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
    GPUtestLOG(sprintf('**** Testing ZEROS (%s,%s)  ',txtfun{f},type1),0);
    if (t==0)
      Ah = zeros(100,100,txtfun{f});
      A = zeros(100,100,feval(gpufun{f}));
    else
      Ah = complex(zeros(100,100,txtfun{f}));
      A = zeros(100,100,complex(feval(gpufun{f})));
      
    end
    
    
    compareCPUGPU(Ah+1,A+1);
    
    if (t==0)
      Ah = zeros(500,100,txtfun{f});
      A = zeros(500,100,feval(gpufun{f}));
    else
      Ah = complex(zeros(500,100,txtfun{f}));
      A = zeros(500,100,complex(feval(gpufun{f})));
      
    end
    
    compareCPUGPU(Ah+1,A+1);
    
    if (t==0)
      Ah = zeros(500,200,100,txtfun{f});
      A = zeros(500,200,100,feval(gpufun{f}));
    else
      Ah = complex(zeros(500,200,100,txtfun{f}));
      A = zeros(500,200,100,complex(feval(gpufun{f})));
      
    end
    
    compareCPUGPU(Ah+1,A+1);
    
    if (t==0)
      Ah = zeros(10,10,2,3,txtfun{f});
      A = zeros(10,10,2,3,feval(gpufun{f}));
    else
      Ah = complex(zeros(10,10,2,3,txtfun{f}));
      A = zeros(10,10,2,3,complex(feval(gpufun{f})));
      
    end
    
    compareCPUGPU(Ah+1,A+1);
    
    if (t==0)
      Ah = zeros(10,10,2,3,4,txtfun{f});
      A = zeros(10,10,2,3,4,feval(gpufun{f}));
    else
      Ah = complex(zeros(10,10,2,3,4,txtfun{f}));
      A = zeros(10,10,2,3,4,complex(feval(gpufun{f})));
      
    end
    
    compareCPUGPU(Ah+1,A+1);
    
    if (t==0)
      Ah = zeros([500,100],txtfun{f});
      A = zeros([500,100],feval(gpufun{f}));
    else
      Ah = complex(zeros([500,100],txtfun{f}));
      A = zeros([500,100],complex(feval(gpufun{f})));
      
    end
    
    compareCPUGPU(Ah+1,A+1);
    
    if (t==0)
      Ah = zeros([10,10,2,3,4,5,6],txtfun{f});
      A = zeros([10,10,2,3,4,5,6],feval(gpufun{f}));
    else
      Ah = complex(zeros([10,10,2,3,4,5,6],txtfun{f}));
      A = zeros([10,10,2,3,4,5,6],complex(feval(gpufun{f})));
      
    end
    
    compareCPUGPU(Ah+1,A+1);
    
  end
end
 GPUtestLOG('***********************************************',0);
end
