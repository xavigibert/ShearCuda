function test_GPUle

config = defaultConfig();
config.optype =4;

op  = 'le';
checkfun(op,config);

end