function test_GPUexp

config = defaultConfig();
config.optype =3;

op  = 'exp';
checkfun(op,config);

end