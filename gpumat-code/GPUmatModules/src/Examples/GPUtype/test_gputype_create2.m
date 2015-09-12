function test_gputype_create2

global GPUtest

try
  type = GPUtest.type;
catch
  error('GPUtest not initialized. Please use GPUtestInit.');
end

%% testmytimes
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
  case 2
    rangei = 0;
  case 3
    rangei = sqrt(-1);
end

testfun = GPUtest.testfun;

for f=1:length(cpufun)
  for i=rangei
    switch i
      case 0
        complexity = 'real';
      case sqrt(-1)
        complexity = 'complex';
    end
    
    
    
    for k=GPUtest.N
      GPUtestLOG(sprintf('**** Testing gputype_create2 (%s,%s) (%d x %d) ',txtfun{f}, complexity,k,k),0);
      
      Am = feval(cpufun{f},feval(testfun,k)+i*feval(testfun,k));
      A = gputype_create2(Am);
      
      compareCPUGPU(Am, A);
      
      
    end
  end
  
end
GPUtestLOG('***********************************************',0);
end
