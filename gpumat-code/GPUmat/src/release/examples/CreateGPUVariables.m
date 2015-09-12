function CreateGPUVariables

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


%% Start GPU
GPUstart

%% Create two arrays on the GPU and initialize them with random numbers
% single precision
A = rand(100,100, GPUsingle); % GPU
B = rand(100,100, GPUsingle); % GPU

%% Create two arrays on the GPU and initialize them with random numbers
% double precision
if (GPUisDoublePrecision)
A = rand(100,100, GPUsingle); % GPU
B = rand(100,100, GPUsingle); % GPU
end

%% Create a vector with zeros using zeros
% single precision
C = zeros(10,10,GPUsingle);
D = zeros([10 10],GPUsingle);
E = zeros(10,10,10,GPUsingle);
F = zeros([10 10 10 10],GPUsingle);

%% Create a vector with zeros using zeros
% double precision
if (GPUisDoublePrecision)
C = zeros(10,10,GPUdouble);
D = zeros([10 10],GPUdouble);
E = zeros(10,10,10,GPUdouble);
F = zeros([10 10 10 10],GPUdouble);
end

%% Create a vector using the colon function
% single precision
G1 = colon(1,10,100,GPUsingle);
H1 = colon(100,-10,1,GPUsingle);
I1 = colon(1,0.1,100,GPUsingle);

%% Create a vector using the colon function
% double precision
if (GPUisDoublePrecision)
G1 = colon(1,10,100,GPUdouble);
H1 = colon(100,-10,1,GPUdouble);
I1 = colon(1,0.1,100,GPUdouble);
end

%% Using G1 and H1 to create a sequence of complex numbers
L  = sqrt(-1) * G1;
M  = sqrt(-1) * pi * 2 * H1;
L0 = complex(G1);

%% Using vertical concatenation
% single precision
N = [zeros(10,1,GPUsingle);colon(1,1,5,GPUsingle)';zeros(10,1,GPUsingle)]; 

%% Using vertical concatenation
% double precision
if (GPUisDoublePrecision)
N = [zeros(10,1,GPUdouble);colon(1,1,5,GPUdouble)';zeros(10,1,GPUdouble)]; 
end

%% Selecting some elements from N using a vector of indexes
idx = GPUsingle([11 12 13]);
O = N(idx);

%% PLEASE NOT THAT ALL ABOVE VARIABLES ARE ON GPU

