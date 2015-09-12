function test_GPUplus

config = defaultConfig();
config.optype =4;

op  = 'plus';
checkfun(op,config);

end