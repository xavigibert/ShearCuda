function test_GPUor

config = defaultConfig();
config.optype =4;
config.type = 2; % real
GPUtestLOG('*** Warning: only REAL', 0);
op  = 'or';

config.filter = 1;
checkfun(op,config);

% filter = 2 means that a logical will be created
config.filter = 0;
checkfun(op,config);

config.filter = 2;
checkfun(op,config);


end