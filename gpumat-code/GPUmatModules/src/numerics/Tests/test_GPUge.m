function test_GPUge

config = defaultConfig();
config.optype =4;

op  = 'ge';
checkfun(op,config);

end