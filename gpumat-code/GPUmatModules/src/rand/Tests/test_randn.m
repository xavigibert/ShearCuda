function test_randn

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
  
  
  % randn with 1 argument
  GPUcompileStart('comp_randn1','-f','-verbose0',a,A)
  R=randn(a,A);
  GPUcompileStop(R)
  
  % randn with 2 argument
  GPUcompileStart('comp_randn2','-f','-verbose0',a,b,A)
  R=randn(a,b,A);
  GPUcompileStop(R)
  
  % randn with 3 argument
  GPUcompileStart('comp_randn3','-f','-verbose0',a,b,c,A)
  R=randn(a,b,c,A);
  GPUcompileStop(R)
  
  % randn with 4 argument
  GPUcompileStart('comp_randn4','-f','-verbose0',a,b,c,d,A)
  R=randn(a,b,c,d,A);
  GPUcompileStop(R)
  
  % randn with 5 argument
  GPUcompileStart('comp_randn5','-f','-verbose0',a,b,c,d,e,A)
  R=randn(a,b,c,d,e,A);
  GPUcompileStop(R)
  
  
  
end

%% testrandn
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
    GPUtestLOG(sprintf('**** Testing randn (%s,%s)  ',txtfun{f},type1),0);
    
    %%
    if (GPUtest.bigKernel)
      Ah = randn(500,200,100,txtfun{f});
      if (GPUtest.checkCompiler==1)
        A = comp_randn3(500,200,100,feval(gpufun{f}));
      else
        A = randn(500,200,100,feval(gpufun{f}));
      end
      
      Ah(:) = feval(cpufun{f},(A(:)));
      err = abs(mean(Ah(:))-0.0);
      if err>0.05
        GPUtestLOG(sprintf('Error in mean tolerance test is %g',err),1);
      end
      err = abs(std(Ah(:))-1.0);
      if err>0.1
        GPUtestLOG(sprintf('Error in stdev tolerance test is %g',err),1);
      end
    end
    
    %% Only even numbers. Check odd
    Ah = randn(5,5,txtfun{f});
    if (GPUtest.checkCompiler==1)
      A = comp_randn2(5,5,feval(gpufun{f}));
    else
      A = randn(5,5,feval(gpufun{f}));
    end
    
    Ah(:) = feval(cpufun{f},(A(:)));
    err = abs(mean(Ah(:))-0.0);
    if err>0.05
      GPUtestLOG(sprintf('Error in mean tolerance test is %g',err),1);
    end
    err = abs(std(Ah(:))-1.0);
    if err>0.1
      GPUtestLOG(sprintf('Error in stdev tolerance test is %g',err),1);
    end
    
    %%
    Ah = randn(100,100,txtfun{f});
    if (GPUtest.checkCompiler==1)
      A = comp_randn2(100,100,feval(gpufun{f}));
    else
      A = randn(100,100,feval(gpufun{f}));
    end
    
    Ah(:) = feval(cpufun{f},(A(:)));
    err = abs(mean(Ah(:))-0.0);
    if err>0.05
      GPUtestLOG(sprintf('Error in mean tolerance test is %g',err),1);
    end
    err = abs(std(Ah(:))-1.0);
    if err>0.1
      GPUtestLOG(sprintf('Error in stdev tolerance test is %g',err),1);
    end
    
    %%
    Ah = randn(500,100,txtfun{f});
    if (GPUtest.checkCompiler==1)
      A = comp_randn2(500,100,feval(gpufun{f}));
    else
      A = randn(500,100,feval(gpufun{f}));
    end
    
    Ah(:) = feval(cpufun{f},(A(:)));
    err = abs(mean(Ah(:))-0.0);
    if err>0.05
      GPUtestLOG(sprintf('Error in mean tolerance test is %g',err),1);
    end
    err = abs(std(Ah(:))-1.0);
    if err>0.1
      GPUtestLOG(sprintf('Error in stdev tolerance test is %g',err),1);
    end
    
    %%
    
    Ah = randn(10,10,2,3,txtfun{f});
    if (GPUtest.checkCompiler==1)
      A = comp_randn4(10,10,2,3,feval(gpufun{f}));
    else
      A = randn(10,10,2,3,feval(gpufun{f}));
    end
    
    Ah(:) = feval(cpufun{f},(A(:)));
    err = abs(mean(Ah(:))-0.0);
    if err>0.05
      GPUtestLOG(sprintf('Error in mean tolerance test is %g',err),1);
    end
    err = abs(std(Ah(:))-1.0);
    if err>0.1
      GPUtestLOG(sprintf('Error in stdev tolerance test is %g',err),1);
    end
    %%
    
    Ah = randn(10,10,2,3,4,txtfun{f});
    if (GPUtest.checkCompiler==1)
      A = comp_randn5(10,10,2,3,4,feval(gpufun{f}));
    else
      A = randn(10,10,2,3,4,feval(gpufun{f}));
    end
    
    Ah(:) = feval(cpufun{f},(A(:)));
    err = abs(mean(Ah(:))-0.0);
    if err>0.05
      GPUtestLOG(sprintf('Error in mean tolerance test is %g',err),1);
    end
    err = abs(std(Ah(:))-1.0);
    if err>0.1
      GPUtestLOG(sprintf('Error in stdev tolerance test is %g',err),1);
    end
    %%
    
    Ah = randn([500,100],txtfun{f});
    if (GPUtest.checkCompiler==1)
      A = comp_randn1([500,100],feval(gpufun{f}));
    else
      A = randn([500,100],feval(gpufun{f}));
    end
    
    Ah(:) = feval(cpufun{f},(A(:)));
    err = abs(mean(Ah(:))-0.0);
    if err>0.05
      GPUtestLOG(sprintf('Error in mean tolerance test is %g',err),1);
    end
    err = abs(std(Ah(:))-1.0);
    if err>0.1
      GPUtestLOG(sprintf('Error in stdev tolerance test is %g',err),1);
    end
    %%
    
    Ah = randn([10,10,2,3,4,5,6],txtfun{f});
    A = randn([10,10,2,3,4,5,6],feval(gpufun{f}));
    
    Ah(:) = feval(cpufun{f},(A(:)));
    err = abs(mean(Ah(:))-0.0);
    if err>0.05
      GPUtestLOG(sprintf('Error in mean tolerance test is %g',err),1);
    end
    err = abs(std(Ah(:))-1.0);
    if err>0.1
      GPUtestLOG(sprintf('Error in stdev tolerance test is %g',err),1);
    end
    
  end
end
GPUtestLOG('***********************************************',0);
end
