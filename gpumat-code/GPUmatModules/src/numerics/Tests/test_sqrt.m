function test_sqrt

config = defaultConfig();
config.optype =1;
config.type = 2; % real
GPUtestLOG('*** Warning: only REAL', 0);
GPUtestLOG('*** Warning: only POSITIVE NUMBERS', 0);
% only positive numbers are generated with rand
config.testfun = @rand;
op  = 'sqrt';
checkfun(op,config);

end