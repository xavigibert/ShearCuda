function testmytimes

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
      GPUtestLOG(sprintf('**** Testing mytimes (%s,%s) (%d x %d) ',txtfun{f}, complexity,k,k),0);
      
      Am = feval(cpufun{f},feval(testfun,k)+i*feval(testfun,k));
      A = feval(gpufun{f},Am);
      Bm = feval(cpufun{f},feval(testfun,k)+i*feval(testfun,k));
      B = feval(gpufun{f},Bm);
      
      % Setting result. If operands are complex, result should be complex
      if (i==0)
        R = zeros(size(A),feval(gpufun{f})); % R is used for the result
      else
        R = complex(zeros(size(A),feval(gpufun{f})));
      end
      
      Rm = Am.*Bm;
      mytimes(A,B,R);
      compareCPUGPU(Rm, R);
      
      
    end
  end
  
end
GPUtestLOG('***********************************************',0);
end
