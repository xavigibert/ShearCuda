function ex1
Ah = rand(1000);           % Matlab variable
A  = GPUsingle(Ah);        % GPU variable
B  = GPUsingle(rand(100)); % GPU variable
C  = GPUdouble(rand(100)); % GPU variable

