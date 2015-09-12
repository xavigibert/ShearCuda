function test_log10

global GPUtest

try
  type = GPUtest.type;
catch
  error('GPUtest not initialized. Please use GPUtestInit.');
end

config = defaultConfig();
config.optype =1;
config.type = 1; % real/complex
GPUtestLOG('*** Warning: only POSITIVE NUMBERS', 0);

% only positive numbers are generated with rand
config.testfun = @rand;

% change accuracy
tol.single = GPUtest.tol.single;
tol.double = GPUtest.tol.double;

GPUtest.tol.single = 7e-5;
GPUtest.tol.double = 4e-15;


op  = 'log10';
checkfun(op,config);


% restore accuracy
GPUtest.tol.single = tol.single;
GPUtest.tol.double = tol.double;
end