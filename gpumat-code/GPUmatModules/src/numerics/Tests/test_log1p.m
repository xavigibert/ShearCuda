function test_log1p

config = defaultConfig();
config.optype =1;
config.type = 2; % real
GPUtestLOG('*** Warning: only REAL', 0);
GPUtestLOG('*** Warning: only POSITIVE NUMBERS', 0);
% only positive numbers are generated with rand
config.testfun = @rand;

op  = 'log1p';
% The error in log10 is bit higher than 1e-15 for double
checkfun(op,config);

end