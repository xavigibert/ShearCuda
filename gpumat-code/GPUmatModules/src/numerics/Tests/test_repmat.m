function test_repmat

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
  R = zeros(5,GPUsingle); % dummy
  a=1; %dummy
  b=2; %dummy
  
  % repmat with 1 argument
  GPUcompileStart('comp_repmat1','-f','-verbose0',A,a)
  R=repmat(A,a);
  GPUcompileStop(R)
  
  % repmat with 2 argument
  GPUcompileStart('comp_repmat2','-f','-verbose0',A,a,b)
  R=repmat(A,a,b);
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
    
    GPUtestLOG(sprintf('**** Testing repmat (%s,%s)',txtfun{f}, complexity),0);
    
    %%
    for k=1:6
      s = fix(rand(1,k)*2+2);
      A = feval(gpufun{f},rand(s)+i*rand(s));
      Ah = feval(cpufun{f},A);
      for l=1:6
        s = fix(rand(1,l)*2+2);
        if (GPUtest.checkCompiler==1)
          B = comp_repmat1(A, s);
        else
          B = repmat(A, s);
        end
        
        Bh = repmat(Ah, s);
        compareCPUGPU(Bh,B);
      end
    end
    
    %%
    % Test bigger matrices
    for u=1:2
      switch u
        case 1
          N = 300;
          M = 400;
          A = feval(gpufun{f},rand(N,M)+i*rand(N,M));
          Ah = feval(cpufun{f},A);
        case 2
          N = 30;
          M = 40;
          K = 10;
          A = feval(gpufun{f},rand(N,M,K)+i*rand(N,M,K));
          Ah = feval(cpufun{f},A);
          
      end
      %%
      for k=1:3
        if (GPUtest.checkCompiler==1)
          B=comp_repmat1(A,k);
        else
          B=repmat(A,k);
        end
        Bh=repmat(Ah,k);
        compareCPUGPU(Bh,B);
      end
      
      %%
      for k=1:3
        if (GPUtest.checkCompiler==1)
          B=comp_repmat2(A,k,k);
        else
          B=repmat(A,k,k);
        end
        Bh=repmat(Ah,k,k);
        compareCPUGPU(Bh,B);
      end
      
      %%
      for k=1:3
        if (GPUtest.checkCompiler==1)
          B=comp_repmat1(A,[k,k]);
        else
          B=repmat(A,[k,k]);
        end
        Bh=repmat(Ah,[k,k]);
        compareCPUGPU(Bh,B);
      end
      
      %%
      for k=1:3
        if (GPUtest.checkCompiler==1)
          B=comp_repmat1(A,[k,k,k]);
        else
          B=repmat(A,[k,k,k]);
        end
        Bh=repmat(Ah,[k,k,k]);
        compareCPUGPU(Bh,B);
      end
      
      %%
      if (GPUtest.checkCompiler==1)
        B=comp_repmat1(A,[2,3,4]);
      else
        B=repmat(A,[2,3,4]);
      end
      Bh=repmat(Ah,[2,3,4]);
      compareCPUGPU(Bh,B);
      
      %%
      if (GPUtest.checkCompiler==1)
        B=comp_repmat1(A,[2,3,4,2]);
      else
        B=repmat(A,[2,3,4,2]);
      end
      Bh=repmat(Ah,[2,3,4,2]);
      compareCPUGPU(Bh,B);
    end
    
    
    
  end
  
  %%
  % test from Forum
  C=GPUsingle(rand(1,1,24));
  Ch = single(C);
  if (GPUtest.checkCompiler==1)
    D=comp_repmat1(C,[100,100,1,12]);
  else
    D=repmat(C,[100,100,1,12]);
  end
  Dh = repmat(Ch,[100,100,1,12]);
  compareCPUGPU(Dh,D);
  
  
  
end
GPUtestLOG('***********************************************',0);
end
