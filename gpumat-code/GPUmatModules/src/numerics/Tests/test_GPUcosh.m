function test_GPUcosh

config = defaultConfig();
config.optype =3;
config.type = 2; % real
GPUtestLOG('*** Warning: only REAL', 0);

op  = 'cosh';
checkfun(op,config);

end
