function test_tanh

config = defaultConfig();
config.optype =1;
config.type = 2; % real
GPUtestLOG('*** Warning: only REAL', 0);

op  = 'tanh';
checkfun(op,config);

end