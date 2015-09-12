function test_GPUrdivide

config = defaultConfig();
config.optype =4;

op  = 'rdivide';
checkfun(op,config);

end