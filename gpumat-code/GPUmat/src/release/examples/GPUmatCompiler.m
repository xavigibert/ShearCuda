function GPUmatCompiler

%
%     Copyright (C) 2012  GP-you Group (http://gp-you.org)
% 
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <http://www.gnu.org/licenses/>.

%% GPUmat compiler

%% Check compiler
GPUcompileCheck

%% Simple compilation
A = randn(5, GPUsingle); % A is a dummy variable

% Compile function C=myexp(B)
GPUcompileStart('myexp','-f',A)
R = exp(A);
GPUcompileStop(R)

% Execute compiled function
B = randn(500, GPUsingle);
F = myexp(B);

%% Two output arguments
A = randn(5, GPUsingle);
B = randn(5, GPUsingle);
% A and B are dummy variables

GPUcompileStart('myfun','-f',A, B)
R1 = exp(A);
R2 = floor(B);
GPUcompileStop(R1,R2)

% Execute compiled function
C = randn(500, GPUsingle);
D = randn(50, GPUsingle);
[T1, T2] = myfun(C,D);

%% Another simple example
A = randn(5, GPUsingle);
% A is a dummy variable

GPUcompileStart('myfun1','-f',A)
R1 = floor(exp(A));
GPUcompileStop(R1)

% Execute compiled function
C = randn(500, GPUsingle);
Z = myfun1(C);

%% For-loop example1
A = randn(5,5,5, GPUsingle);
B = randn(5, GPUsingle);
GPUcompileStart('myfor1', '-f',  A, B)
GPUfor it=1:5
  assign(1,A,B,':',':',it)
GPUend
GPUcompileStop

% Execute compiled function
A = randn(50,50,5, GPUsingle);
B = randn(50, GPUsingle);
myfor1(A, B);

%% For-loop example2
A = randn(5,5,5, GPUsingle);
B = randn(1,5, GPUsingle);
GPUcompileStart('myfor2', '-f', A, B)
GPUfor it=1:5
  GPUfor jt=1:5
    assign(1,A,B,':',jt,it)
  GPUend
GPUend
GPUcompileStop

% Execute compiled function
A = randn(50,50,5, GPUsingle);
B = randn(1,50, GPUsingle);
myfor2(A, B);

%% Matlab variables as input
A = randn(5,5, GPUsingle);
a = 1; % dummy
b = 1; % dummy
c = 1; % dummy
GPUcompileStart('code_ex3', '-f', A, a, b, c)
assign(1,A,a,b,c)
GPUcompileStop

% Execute compiled function
A = randn(3,3, GPUsingle)
code_ex3(A,single(2),':',':')
A

A = randn(30,30, GPUsingle)
code_ex3(A,single(4),':',':')
A

A = randn(30,30, GPUsingle)
code_ex3(A,single(4),':',1)
A

%% Another example
N = 15;
kernel_fft_rot = rand(N,N,4,5, GPUsingle);
vis_fft = rand(N,N,4, GPUsingle);
C1 = zeros(N,N,4,GPUsingle);

P = size(kernel_fft_rot,4);
M = size(vis_fft,3);
GPUcompileStart('forloop1', '-f', kernel_fft_rot, vis_fft, C1)
GPUfor it=1:P
  GPUfor jt=1:M
    out= real(ifft2(slice(vis_fft,':',':',jt).*slice(kernel_fft_rot,':',':',jt,it)));
    tmp1 = slice(C1,':',':',jt) + out;
    assign(1,C1, tmp1, ':',':',jt)
  GPUend
GPUend
GPUcompileStop

% now run code on GPUmat
kernel_fft_rot = rand(N,N,4,5, GPUsingle);
vis_fft = rand(N,N,4, GPUsingle);
C1 = zeros(N,N,4,GPUsingle);
C2 = zeros(N,N,4,GPUsingle);

P = size(kernel_fft_rot,4);
M = size(vis_fft,3);
for it=1:P
  for jt=1:M
    out= real(ifft2(slice(vis_fft,':',':',jt).*slice(kernel_fft_rot,':',':',jt,it)));
    tmp1 = slice(C1,':',':',jt) + out;
    assign(1,C1, tmp1, ':',':',jt)
  end
end

% same with compiled function
forloop1(kernel_fft_rot, vis_fft, C2);

compareCPUGPU(single(C2),C1);



end


