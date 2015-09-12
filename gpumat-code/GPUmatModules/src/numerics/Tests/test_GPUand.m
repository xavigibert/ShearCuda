function test_GPUand

config = defaultConfig();
config.optype =4;
config.type = 2; % real
GPUtestLOG('*** Warning: only REAL', 0);
op  = 'and';

config.filter = 1;
checkfun(op,config);

% filterB and filterA == 2 means that a logical will be created
config.filter = 0;
checkfun(op,config);

config.filter = 2;
checkfun(op,config);


end
