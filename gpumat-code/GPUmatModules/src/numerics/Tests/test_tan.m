function test_tan

config = defaultConfig();
config.optype =1;
config.type = 2; % real
GPUtestLOG('*** Warning: only REAL', 0);

op  = 'tan';
checkfun(op,config);

end