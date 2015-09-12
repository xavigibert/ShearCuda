function test_sinh

config = defaultConfig();
config.optype =1;
config.type = 2; % real
GPUtestLOG('*** Warning: only REAL', 0);

op  = 'sinh';
checkfun(op,config);

end