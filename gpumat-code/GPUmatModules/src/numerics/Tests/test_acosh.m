function test_acosh

config = defaultConfig();
config.optype =1;
config.type = 2; % real
GPUtestLOG('*** Warning: only REAL', 0);

% only [1:pi] 
config.testfun = @rand_acosh;

op  = 'acosh';
checkfun(op,config);

end
