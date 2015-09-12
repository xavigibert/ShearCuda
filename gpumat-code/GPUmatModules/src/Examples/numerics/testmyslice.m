function testmyslice

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
    
    
    k = 30;
    GPUtestLOG(sprintf('**** Testing myslice1 (%s,%s) (%d x %d x %d) ',txtfun{f}, complexity,k,k,k),0);
    
    Am = feval(cpufun{f},feval(testfun,k,k,k)+i*feval(testfun,k,k,k));
    A = feval(gpufun{f},Am);
    Bm = feval(cpufun{f},feval(testfun,k,k,k)+i*feval(testfun,k,k,k));
    B = feval(gpufun{f},Bm);
    
    % The index to functions myslice1 and myslice2 are given in Matlab
    % format, with first element = 1
    
    % myslice1 used with last parameter set to 0 assigns 
    % A = B(range)
    % myslice1 used with last parameter set to 1 assigns 
    % A(range) = B
    
    % different tests with different ranges
    myslice1(A, B, 1,1,k,1,1,k,1,1,k,0);
    Rm = Bm(1:end,1:end,1:end); % equivalent to above command
    compareCPUGPU(Rm, A);
    
    % same as above with myslice2
    R = myslice2(B, 1,1,k,1,1,k,1,1,k);
    Rm = Bm(1:end,1:end,1:end); % equivalent to above command
    compareCPUGPU(Rm, R);
    
    Am = feval(cpufun{f},feval(testfun,1,k,k)+i*feval(testfun,1,k,k));
    A = feval(gpufun{f},Am);
    Bm = feval(cpufun{f},feval(testfun,k,k,k)+i*feval(testfun,k,k,k));
    B = feval(gpufun{f},Bm);
    
    % different tests with different ranges
    myslice1(A, B, 1,0,0,1,1,k,1,1,k,0);
    Rm = Bm(1,1:end,1:end); % equivalent to above command
    compareCPUGPU(Rm, A);
    
    % same as above with myslice2
    R = myslice2(B, 1,0,0,1,1,k,1,1,k);
    Rm = Bm(1,1:end,1:end); % equivalent to above command
    compareCPUGPU(Rm, R);
    
    
    
    
    
  end
  
end
GPUtestLOG('***********************************************',0);
end
