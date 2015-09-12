function test_GPUtan

config = defaultConfig();
config.optype =3;
config.type = 2; % real
GPUtestLOG('*** Warning: only REAL', 0);

op  = 'tan';
checkfun(op,config);

end