function test_GPUsinh

config = defaultConfig();
config.optype =3;
config.type = 2; % real
GPUtestLOG('*** Warning: only REAL', 0);

op  = 'sinh';
checkfun(op,config);

end