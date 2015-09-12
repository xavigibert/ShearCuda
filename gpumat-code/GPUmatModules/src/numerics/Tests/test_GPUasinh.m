function test_GPUasinh

config = defaultConfig();
config.optype =3;
config.type = 2; % real
GPUtestLOG('*** Warning: only REAL', 0);

op  = 'asinh';
checkfun(op,config);

end
