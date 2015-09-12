function test_transpose

global GPUtest

try
  type = GPUtest.type;
catch
  error('GPUtest not initialized. Please use GPUtestInit.');
end

config = defaultConfig();
config.optype =1;


op  = 'transpose';
checkfun(op,config);

%% additional test
if (GPUtest.fastMode)
  return
end

gpufun = GPUtest.gpufun;
cpufun = GPUtest.cpufun;

type = GPUtest.type;
if (type == 1)
  rangei = [0 sqrt(-1)];
  
end

if (type == 2)
  rangei = 0;
  
end

if (type == 3)
  rangei = [0 sqrt(-1)];
  rangej = [0];
end

%% Test operations with different types
for i=rangei
  for f=1:length(cpufun)
    A = feval(gpufun{f}, rand(1,2e6)+i*rand(1,2e6));
    Am = feval(cpufun{f},A);
    
    E = A.';
    Em = Am.';
    
    clear A
    clear Am
    compareCPUGPU(Em, E);
    
    A = feval(gpufun{f}, rand(2e6,1)+i*rand(2e6,1));
    Am = feval(cpufun{f},A);
    
    E = A.';
    Em = Am.';
    
    clear A
    clear Am
    compareCPUGPU(Em, E);
    
    A = feval(gpufun{f}, rand(2e6,5)+i*rand(2e6,5));
    Am = feval(cpufun{f},A);
    
    E = A.';
    Em = Am.';
    
    clear A
    clear Am
    compareCPUGPU(Em, E);
  end
end
end