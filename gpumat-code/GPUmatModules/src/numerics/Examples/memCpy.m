function memCpy

%% Host -> Device copy

% Copy 20 elements from X to R(100)
R = GPUsingle(rand(100,100));
X = single(rand(100,100));
memCpyHtoD(R, X, 100, 20)

%% Device -> Device copy


% Copy 20 elements from X to R(100)
R = GPUsingle(rand(100,100));
X = GPUsingle(rand(100,100));
memCpyDtoD(R, X, 100, 20)





