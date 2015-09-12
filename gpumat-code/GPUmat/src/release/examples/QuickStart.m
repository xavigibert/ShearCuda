function QuickStart

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

%% Create two arrays on the GPU 
%  Initialize them with random numbers
% single precision
A = rand(100,100, GPUsingle); % A is on GPU memory
B = rand(100,100, GPUsingle); % B is on GPU memory

% double precision
if (GPUisDoublePrecision)
A = rand(100,100, GPUdouble); % A is on GPU memory
B = rand(100,100, GPUdouble); % B is on GPU memory
end

%% Add A and B, store the result in array C
C = A+B; % executed on GPU

%% Scalars are automatically converted
C = A+1; % Matlab scalar doesn't need conversion
         % to GPU variable

%% Convert GPU variables into Matlab variables
% single precision
Ah = single(A); % Ah is on HOST memory
Bh = single(B); % Bh is on HOST memory
Ch = single(C); % Ch is on HOST memory

% double precision
if (GPUisDoublePrecision)
Ah = double(A); % Ah is on HOST memory
Bh = double(B); % Bh is on HOST memory
Ch = double(C); % Ch is on HOST memory
end

%% Multiply A and B
C = A*B; % executed on GPU

%% Convert GPU variables into Matlab variables
Ch = single(C); % Ch is on HOST memory

%% Some more complicated operation
C = exp(A)*B + B.^2; % executed on GPU

%% Convert GPU variables into Matlab variables
Ch = single(C); % Ch is on HOST memory

%% FFT transform 

%% Create two arrays on the GPU 
%  Initialize them with random numbers
A = rand(1,100, GPUsingle);   % A is on GPU memory
B = rand(100,100, GPUsingle); % B is on GPU memory

%% 1D FFT
FFT_A = fft(A); % executed on GPU

%% 2D FFT
FFT_B = fft2(B); % executed on GPU

%% Convert GPU variables into Matlab variables
FFT_Ah = single(FFT_A); % FFT_Ah is on HOST
FFT_Bh = single(FFT_B); % FFT_Bh is on HOST





