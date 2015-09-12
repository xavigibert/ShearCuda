function test_GPUlt

config = defaultConfig();
config.optype =4;

op  = 'lt';
checkfun(op,config);

end