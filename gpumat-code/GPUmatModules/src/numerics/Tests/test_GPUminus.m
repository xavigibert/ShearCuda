function test_GPUminus

config = defaultConfig();
config.optype =4;

op  = 'minus';
checkfun(op,config);

end