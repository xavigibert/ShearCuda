function test_ctranspose

config = defaultConfig();
config.optype =1;

op  = 'ctranspose';
checkfun(op,config);

end
