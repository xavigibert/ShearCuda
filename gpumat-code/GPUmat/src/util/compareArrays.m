function compareArrays(h_C_ref, h_C, epsilon)
% compareArrays Numerically compare two arrays

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