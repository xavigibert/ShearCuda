function r = int32(g)
% int32 - Converts a GPU variable into a Matlab int32 precision
% variable
%
% SYNTAX
%
% R = int32(X)
% X - GPUtype
% R - Matlab variable
%
%
% DESCRIPTION
% B = INT32(A) returns the contents of the GPU variable A into a
% int32 precision Matlab array.
%
% EXAMPLE
%
% A = GPUint32(rand(100))
% Ah = int32(A);

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


if (GPUstart(1)==0)
  error('GPU not started. Please start the GPU using GPUstart');
end


if (g.GPUtypePtr == 0 || isempty(g))
  r = [];
  return;
end

gpuFLOAT   = 0;
gpuCFLOAT  = 1;
gpuDOUBLE  = 2;
gpuCDOUBLE = 3;
gpuINT32   = 4;
type = getType(g);

switch type
  case {gpuFLOAT , gpuCFLOAT}
    A = GPUtypeToSingle(g);
  case {gpuDOUBLE , gpuCDOUBLE}
    A = GPUtypeToDouble(g);
  case gpuINT32
    A = GPUtypeToInt32(g);
end

% should reshape and handle complex


s = size(g);

switch type
  case {gpuCDOUBLE , gpuCFLOAT}
    imR = zeros(s);
    reR = zeros(s);
    
    reR(1:end) = A(1:2:end);
    imR(1:end) = A(2:2:end);
    r = int32(reR+ sqrt(-1)*imR);
  case {gpuDOUBLE , gpuFLOAT, gpuINT32 }
    r = int32(A);
end


r = reshape(r,s);

end

