function test_GPUuminus

config = defaultConfig();
config.optype =3;


op  = 'uminus';
checkfun(op,config);

end