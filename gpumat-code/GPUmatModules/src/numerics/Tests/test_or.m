function test_or

config = defaultConfig();
config.optype =2;
config.type = 2; % real
GPUtestLOG('*** Warning: only REAL', 0);
op  = '|';

config.filter = 1;
checkfun(op,config);

% filter = 2 means that a logical will be created
config.filter = 0;
checkfun(op,config);

config.filter = 2;
checkfun(op,config);


end