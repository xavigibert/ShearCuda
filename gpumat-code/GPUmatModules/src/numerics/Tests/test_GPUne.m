function test_GPUne

config = defaultConfig();
config.optype =4;
config.filter = 1;

op  = 'ne';
checkfun(op,config);

end