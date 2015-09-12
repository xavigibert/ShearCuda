function test_cosh

config = defaultConfig();
config.optype =1;
config.type = 2; % real
GPUtestLOG('*** Warning: only REAL', 0);

op  = 'cosh';
checkfun(op,config);

end
