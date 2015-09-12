function test_GPUfloor

config = defaultConfig();
config.optype =3;
config.type = 2; % real
GPUtestLOG('*** Warning: only REAL', 0);

op  = 'floor';
checkfun(op,config);

end