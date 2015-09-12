function test_GPUconj

config = defaultConfig();
config.optype =3;

op  = 'conj';
checkfun(op,config);

end
