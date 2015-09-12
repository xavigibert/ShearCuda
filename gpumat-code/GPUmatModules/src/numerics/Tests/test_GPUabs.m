function test_GPUabs

config = defaultConfig();
config.optype =3;

op  = 'abs';
checkfun(op,config);

end
