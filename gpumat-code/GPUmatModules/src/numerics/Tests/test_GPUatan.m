function test_GPUatan

config = defaultConfig();
config.optype =3;
config.type = 2; % real
GPUtestLOG('*** Warning: only REAL', 0);

op  = 'atan';
checkfun(op,config);

end
