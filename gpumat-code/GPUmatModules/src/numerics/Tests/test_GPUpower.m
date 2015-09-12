function test_GPUpower

global GPUtest

try
  type = GPUtest.type;
catch
  error('GPUtest not initialized. Please use GPUtestInit.');
end

config = defaultConfig();
config.optype =4;
%config.type =3;

tol.single = GPUtest.tol.single;
tol.double = GPUtest.tol.double;

GPUtest.tol.single = 5e-6;
GPUtest.tol.double = 5e-15;

% only positive 
GPUtestLOG('*** Warning: only POSITIVE', 0);
config.testfun = @rand;

op  = 'power';
checkfun(op,config);

GPUtest.tol.single = tol.single;
GPUtest.tol.double =tol.double;

end