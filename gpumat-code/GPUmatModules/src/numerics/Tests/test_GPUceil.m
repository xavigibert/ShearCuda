function test_GPUceil

config = defaultConfig();
config.optype =3;
config.type = 2; % real
GPUtestLOG('*** Warning: only REAL', 0);

op  = 'ceil';
checkfun(op,config);

end
