function test_GPUfill

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
  R = zeros(5,GPUsingle); % dummy
  a=1; %dummy
  b=2; %dummy
  c=3; %dummy
  d=4; %dummy
  e=5; %dummy
  f=6; %dummy
  GPUcompileStart('comp_GPUfill','-f','-verbose0',R,a,b,c,d,e,f)
  GPUfill(R,a,b,c,d,e,f);
  GPUcompileStop
end
%% Test memCpyDtoD
for f=1:length(cpufun)
  
  
  
  for i=rangei
    switch i
      case 0
        complexity = 'real';
      case sqrt(-1)
        complexity = 'complex';
    end
    
    
    GPUtestLOG(sprintf('**** Testing GPUfill (%s,%s)',txtfun{f}, complexity),0);
    N = 100;
    for type=0:2
      %%
      for k=1:10000:100000
        A = feval(gpufun{f},rand(N,N)+i*rand(N,N));
        Ah = feval(cpufun{f},A);
        
        if (GPUtest.checkCompiler==1)
          comp_GPUfill(A,k,0.0,1,1,0,type);
        else
          GPUfill(A,k,0.0,1,1,0,type);
        end
        
        Ah = mxGPUfill(Ah,k,0.0,1,1,0,type);
        compareCPUGPU(Ah,A);
      end
      %%
      A = feval(gpufun{f},rand(N,N)+i*rand(N,N));
      Ah = feval(cpufun{f},A);
      
      if (GPUtest.checkCompiler==1)
        comp_GPUfill(A,-1.0,0.0,1,1,0,type);
      else
        GPUfill(A,-1.0,0.0,1,1,0,type);
      end
      Ah = mxGPUfill(Ah,-1.0,0.0,1,1,0,type);
      
      compareCPUGPU(Ah,A);
      
      %%
      for k=0:20:100
        A = feval(gpufun{f},rand(N,N)+i*rand(N,N));
        Ah = feval(cpufun{f},A);
        
        if (GPUtest.checkCompiler==1)
          comp_GPUfill(A,1.0,k,1,1,0,type);
        else
          GPUfill(A,1.0,k,1,1,0,type);
        end
        Ah = mxGPUfill(Ah,1.0,k,1,1,0,type);
        
        compareCPUGPU(Ah,A);
      end
      
      %%
      for k=0:20:100
        A = feval(gpufun{f},rand(N,N)+i*rand(N,N));
        Ah = feval(cpufun{f},A);
        
        if (GPUtest.checkCompiler==1)
          comp_GPUfill(A,0.5,0.5,k,1,0,type);
        else
          GPUfill(A,0.5,0.5,k,1,0,type);
        end
        Ah = mxGPUfill(Ah,0.5,0.5,k,1,0,type);
        
        compareCPUGPU(Ah,A);
      end
      
      
      %%
      for k=0:10:100
        for l=0:3:20
          
          A = feval(gpufun{f},rand(N,N)+i*rand(N,N));
          Ah = feval(cpufun{f},A);
          %k = 0;
          if (GPUtest.checkCompiler==1)
            comp_GPUfill(A,0.5,0.5,10,k,l,0);
          else
            GPUfill(A,0.5,0.5,10,k,l,0);
          end
          Ah = mxGPUfill(Ah,0.5,0.5,10,k,l,0);
          
          compareCPUGPU(Ah,A);
        end
      end
      
      
      
      
      
    end
    
  end
  
end


GPUtestLOG('***********************************************',0);
end

function A = mxGPUfill(A,offset,incr,m,p,offsp,type)
if (m<=0)
  m = numel(A);
end

if (p<=0)
  p = 1;
end


xIndex = (1:numel(A))-1;
index = mod((1:numel(A))-1+offsp,p)==0;
c = incr*(mod(xIndex,m)) + offset;

if (~isreal(A))
  re = real(A);
  im = imag(A);
  switch type
    case 0
      re(index) = c(index);
    case 1
      im(index) = c(index);
    case 2
      re(index) = c(index);
      im(index) = c(index);
  end
  A = re +sqrt(-1)*im;
else
  A(index) = c(index);
end

end

