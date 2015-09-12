function test_GPUgt

config = defaultConfig();
config.optype =4;

op  = 'gt';
checkfun(op,config);

end