function compareCPUGPU( cpu, gpu)
% compareCPUGPU compares CPU and GPU

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

% compareClass returns the handle to the function that shoud be used to
% compare GPU and CPU variables. Returns also the tolerance
[compfun,tol] = compareClass(cpu,gpu);

% numerically compare arrays
compareArrays(cpu,feval(compfun,gpu),tol);

% avoid the following condition
% cpu =
%   Empty matrix: 1-by-0
% gpu
%   []
% size(cpu) = 1 0
% size(gpu) = 0 0

if (isempty(cpu)&&isempty(gpu))
  return
end

% compare also the size of GPU and CPU variables
% remove ones
scpu = size(cpu);
nscpu = length(scpu);
for i=length(scpu):-1:1
  if scpu(i) == 1
    nscpu = i;
  else
    break
  end
end
scpu = scpu(1:nscpu);

sgpu = size(gpu);
nsgpu = length(sgpu);
for i=length(sgpu):-1:1
  if sgpu(i) == 1
    nsgpu = i;
  else
    break
  end
end
sgpu = scpu(1:nsgpu);

%compareArrays(size(cpu), size(gpu),1e-6);
compareArrays(scpu, sgpu,1e-6);


end


function compareArrays(h_C_ref, h_C, epsilon)
% compareArrays Numerically compare two arrays

global GPUtest

try
  type = GPUtest.type;
catch
  error('GPUtest not initialized. Please use GPUtestInit.');
end


%%h_C_ref = h_C_ref(1:end);
%%h_C = h_C(1:end);

if (numel(h_C_ref)~=numel(h_C))
  GPUtestLOG('Number of elements is different',1);
  return;
end

% check if both empty
if (isempty(h_C_ref) && isempty(h_C))
  return
end

% compared array has only zero elements. Generates error or just warning
% depending on GPUtest.noCompareZeros
if (nnz(h_C_ref)==0)
  GPUtestLOG('Warning: Compared arrays are zeros',GPUtest.noCompareZeros);
end

if (nnz(h_C)==0)
  GPUtestLOG('Warning: Compared arrays are zeros',GPUtest.noCompareZeros);
end


idx_nonzero = find(abs(h_C_ref));
idx_zero = find(~abs(h_C_ref));

ref_error_nonzero = 0;
ref_error_zero = 0;

if (~isempty(idx_nonzero))
  ref_error_nonzero = max(abs((h_C_ref(idx_nonzero) - h_C(idx_nonzero))./h_C_ref(idx_nonzero)));
end
if (~isempty(idx_zero))
  ref_error_zero = max(abs(h_C_ref(idx_zero) - h_C(idx_zero)));
end

ref_error = max(ref_error_nonzero,ref_error_zero);
if (ref_error < epsilon)
else
  GPUtestLOG (['Arrays are different (error is ' num2str(ref_error) ')!!!'], 1);
end


%GPUtestLOG(' ',0);

end

function [ compfun, tol ] = compareClass(cpu, gpu)
%COMPARECLASS Compare CPU and GPU class
%   Compare CPU and GPU class. Returns the handle that should be used to
%   compare GPU and CPU object, and the tolerance to be used

global GPUtest

try
  type = GPUtest.type;
catch
  error('GPUtest not initialized. Please use GPUtestInit.');
end

% check if both are complex
if isreal(cpu)
  if (~isreal(gpu))
    GPUtestLOG('Error, CPU real and GPU complex',1);
  end
end
if ~isreal(cpu)
  if (isreal(gpu))
    GPUtestLOG('Error, CPU complex and GPU real',1);
  end
end

% check class
if isa(cpu,'single')
  compfun = @single;
  tol = GPUtest.tol.single;
  if ~isa(gpu,'GPUsingle')
    if (isscalar(gpu))
      % nothing. 
    else
      GPUtestLOG('Error, expected GPUsingle',1);
    end
    
  end
  
elseif isa(cpu,'double')
  compfun = @double;
  tol = GPUtest.tol.double;
  if ~isa(gpu,'GPUdouble')
    if (isscalar(gpu))
      % nothing
    else
      GPUtestLOG('Error, expected GPUdouble',1);
    end
  end
  % logical not implemented, GPUsingle returned
elseif isa(cpu,'logical')
  compfun = @single;
  tol = GPUtest.tol.single;
  if ~isa(gpu,'GPUsingle')
    GPUtestLOG('Error, expected GPUsingle',1);
  end
else
  error('Unknown class');
end

end




