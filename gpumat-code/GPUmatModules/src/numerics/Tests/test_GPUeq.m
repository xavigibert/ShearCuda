function test_GPUeq

config = defaultConfig();
config.optype =4;
config.filter = 1;

op  = 'eq';
checkfun(op,config);

end
