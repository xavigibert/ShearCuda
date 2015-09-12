function test_cos

config = defaultConfig();
config.optype =1;
config.type = 2; % real
GPUtestLOG('*** Warning: only REAL', 0);

% only [-1:1] 
config.testfun = @rand_plusminusone;

op  = 'cos';
checkfun(op,config);

end
