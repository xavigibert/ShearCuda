function test_GPUasin

config = defaultConfig();
config.optype =3;
config.type = 2; % real
GPUtestLOG('*** Warning: only REAL', 0);

% only [-1:1] 
config.testfun = @rand_plusminusone;

op  = 'asin';
checkfun(op,config);

end
