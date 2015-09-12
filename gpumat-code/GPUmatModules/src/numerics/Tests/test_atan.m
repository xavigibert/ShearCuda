function test_atan

config = defaultConfig();
config.optype =1;
config.type = 2; % real
GPUtestLOG('*** Warning: only REAL', 0);

op  = 'atan';
checkfun(op,config);

end
