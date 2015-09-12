function test_rand

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
  
  
  % rand with 1 argument
  GPUcompileStart('comp_rand1','-f','-verbose0',a,A)
  R=rand(a,A);
  GPUcompileStop(R)
  
  % rand with 2 argument
  GPUcompileStart('comp_rand2','-f','-verbose0',a,b,A)
  R=rand(a,b,A);
  GPUcompileStop(R)
  
  % rand with 3 argument
  GPUcompileStart('comp_rand3','-f','-verbose0',a,b,c,A)
  R=rand(a,b,c,A);
  GPUcompileStop(R)
  
  % rand with 4 argument
  GPUcompileStart('comp_rand4','-f','-verbose0',a,b,c,d,A)
  R=rand(a,b,c,d,A);
  GPUcompileStop(R)
  
  % rand with 5 argument
  GPUcompileStart('comp_rand5','-f','-verbose0',a,b,c,d,e,A)
  R=rand(a,b,c,d,e,A);
  GPUcompileStop(R)
  
  
  
end

%% testrand
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
    GPUtestLOG(sprintf('**** Testing RAND (%s,%s)  ',txtfun{f},type1),0);
    
    %%
    if (GPUtest.bigKernel)
      Ah = rand(500,200,100,txtfun{f});
      if (GPUtest.checkCompiler==1)
        A = comp_rand3(500,200,100,feval(gpufun{f}));
      else
        A = rand(500,200,100,feval(gpufun{f}));
      end
      
      Ah(:) = feval(cpufun{f},(A(:)));
      err = abs(mean(Ah(:))-0.5);
      if err>0.05
        GPUtestLOG(sprintf('Error in tolerance test is %g',err),1);
      end
    end
    
    %%
    Ah = rand(100,100,txtfun{f});
    if (GPUtest.checkCompiler==1)
      A = comp_rand2(100,100,feval(gpufun{f}));
    else
      A = rand(100,100,feval(gpufun{f}));
    end
    
    Ah(:) = feval(cpufun{f},(A(:)));
    err = abs(mean(Ah(:))-0.5);
    if err>0.05
      GPUtestLOG(sprintf('Error in tolerance test is %g',err),1);
    end
    
    %%
    Ah = rand(500,100,txtfun{f});
    if (GPUtest.checkCompiler==1)
      A = comp_rand2(500,100,feval(gpufun{f}));
    else
      A = rand(500,100,feval(gpufun{f}));
    end
    
    Ah(:) = feval(cpufun{f},(A(:)));
    err = abs(mean(Ah(:))-0.5);
    if err>0.05
      GPUtestLOG(sprintf('Error in tolerance test is %g',err),1);
    end
    
    %%
    
    Ah = rand(10,10,2,3,txtfun{f});
    if (GPUtest.checkCompiler==1)
      A = comp_rand4(10,10,2,3,feval(gpufun{f}));
    else
      A = rand(10,10,2,3,feval(gpufun{f}));
    end
    
    Ah(:) = feval(cpufun{f},(A(:)));
    err = abs(mean(Ah(:))-0.5);
    if err>0.05
      GPUtestLOG(sprintf('Error in tolerance test is %g',err),1);
    end
    %%
    
    Ah = rand(10,10,2,3,4,txtfun{f});
    if (GPUtest.checkCompiler==1)
      A = comp_rand5(10,10,2,3,4,feval(gpufun{f}));
    else
      A = rand(10,10,2,3,4,feval(gpufun{f}));
    end
    
    Ah(:) = feval(cpufun{f},(A(:)));
    err = abs(mean(Ah(:))-0.5);
    if err>0.05
      GPUtestLOG(sprintf('Error in tolerance test is %g',err),1);
    end
    %%
    
    Ah = rand([500,100],txtfun{f});
    if (GPUtest.checkCompiler==1)
      A = comp_rand1([500,100],feval(gpufun{f}));
    else
      A = rand([500,100],feval(gpufun{f}));
    end
    
    Ah(:) = feval(cpufun{f},(A(:)));
    err = abs(mean(Ah(:))-0.5);
    if err>0.05
      GPUtestLOG(sprintf('Error in tolerance test is %g',err),1);
    end
    %%
    
    Ah = rand([10,10,2,3,4,5,6],txtfun{f});
    A = rand([10,10,2,3,4,5,6],feval(gpufun{f}));
    
    Ah(:) = feval(cpufun{f},(A(:)));
    err = abs(mean(Ah(:))-0.5);
    if err>0.05
      GPUtestLOG(sprintf('Error in tolerance test is %g',err),1);
    end
    
  end
end
GPUtestLOG('***********************************************',0);
end
