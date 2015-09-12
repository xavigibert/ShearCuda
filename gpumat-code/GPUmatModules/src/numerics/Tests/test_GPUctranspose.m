function test_GPUctranspose

config = defaultConfig();
config.optype =3;

op  = 'ctranspose';
checkfun(op,config);

end
