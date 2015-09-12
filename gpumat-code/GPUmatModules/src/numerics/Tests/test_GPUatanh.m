function test_GPUatanh

config = defaultConfig();
config.optype =3;
config.type = 2; % real
GPUtestLOG('*** Warning: only REAL', 0);

% only [-1:1] 
config.testfun = @rand_plusminusone;

op  = 'atanh';
checkfun(op,config);

end
