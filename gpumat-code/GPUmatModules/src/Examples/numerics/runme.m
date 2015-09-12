function runme
%% run test
disp('* Start test');
A = GPUsingle(rand(100));
B = GPUsingle(rand(100));
R = zeros(size(A), GPUsingle);
myplus(A,B,R);
mytimes(A,B,R);
disp('* Test finished');
end