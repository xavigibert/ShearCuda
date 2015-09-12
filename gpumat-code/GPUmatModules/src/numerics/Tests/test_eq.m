function test_eq

config = defaultConfig();
config.optype =2;
config.filter = 1;

op  = '==';
checkfun(op,config);

end
