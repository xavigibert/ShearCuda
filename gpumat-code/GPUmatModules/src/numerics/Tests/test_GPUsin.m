function test_GPUsin

config = defaultConfig();
config.optype =3;
config.type = 2; % real
GPUtestLOG('*** Warning: only REAL', 0);

op  = 'sin';
checkfun(op,config);

end