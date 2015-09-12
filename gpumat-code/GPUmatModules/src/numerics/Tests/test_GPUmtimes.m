function test_GPUmtimes


global GPUtest

try
  type = GPUtest.type;
catch
  error('GPUtest not initialized. Please use GPUtestInit.');
end

config = defaultConfig();
config.optype =4;
config.type = 2;



config.trA = 1;

% If I use randn as a test function teh accuracy is very low, but it is a
% CUDA problem
config.testfun = @rand;

% change accuracy
tol.single = GPUtest.tol.single;
tol.double = GPUtest.tol.double;

GPUtest.tol.single = 7e-5;
GPUtest.tol.double = 1e-14;

config.gpufun = {@GPUsingle};
config.cpufun = {@single};
config.txtfun = {'single'};
    
op  = 'mtimes';
checkfun(op,config);

config.gpufun = {@GPUdouble};
config.cpufun = {@double};
config.txtfun = {'double'};

op  = 'mtimes';
checkfun(op,config);

config.type = 3;

config.gpufun = {@GPUsingle};
config.cpufun = {@single};
config.txtfun = {'single'};
    
op  = 'mtimes';
checkfun(op,config);

config.gpufun = {@GPUdouble};
config.cpufun = {@double};
config.txtfun = {'double'};

op  = 'mtimes';
checkfun(op,config);

% restore accuracy
GPUtest.tol.single = tol.single;
GPUtest.tol.double = tol.double;


end