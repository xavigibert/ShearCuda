function test_GPUldivide

config = defaultConfig();
config.optype =4;

op  = 'ldivide';
checkfun(op,config);

end