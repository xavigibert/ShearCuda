function config = defaultConfig()

% If config.type>=0 override global GPUtest.type
config.type = -1;

% optype
% 1 - op(A)
% 2 - A op B
% 3 - op(A,B)
config.optype =1;

% create A or B with elements transposed
config.trA = 0;
config.trB = 0;

% filter input matrix
config.filter = 0;

% test function
config.testfun = -1;

% gpufun and cpufun
config.gpufun = -1;
config.cpufun = -1;
config.txtfun = -1;


end
