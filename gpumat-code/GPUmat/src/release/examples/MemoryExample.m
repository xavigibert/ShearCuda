function MemoryExample

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

%% Example of GPU memory clean up

%% Print available GPU memory  
disp('Memory before allocation (bytes)');
GPUmem

%% Allocate variable
A = GPUsingle(rand(1,5e6));

GPUmem
disp('Memory after allocation (bytes)');

%% delete variable
clear A;

%% Print available GPU memory  
GPUmem
disp('Memory after clear (bytes)');
