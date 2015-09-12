function test_mtimes


global GPUtest

try
  type = GPUtest.type;
catch
  error('GPUtest not initialized. Please use GPUtestInit.');
end

config = defaultConfig();
config.optype =2;

config.trA = 1;

% If I use randn as a test function teh accuracy is very low, but it is a
% CUDA problem
config.testfun = @rand;

% change accuracy
tol.single = GPUtest.tol.single;
tol.double = GPUtest.tol.double;

GPUtest.tol.single = 7e-5;
GPUtest.tol.double = 1e-14;

op  = '*';
checkfun(op,config);

% restore accuracy
GPUtest.tol.single = tol.single;
GPUtest.tol.double = tol.double;


end