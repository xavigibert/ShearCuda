function test_GPUlog2

config = defaultConfig();
config.optype =3;
config.type = 2; % real
GPUtestLOG('*** Warning: only REAL', 0);
GPUtestLOG('*** Warning: only POSITIVE NUMBERS', 0);
% only positive numbers are generated with rand
config.testfun = @rand;

op  = 'log2';
checkfun(op,config);

end