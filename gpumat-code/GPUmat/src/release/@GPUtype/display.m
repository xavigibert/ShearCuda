function display(p)
% display - Display GPU variable
%
% SYNTAX
%
% display(X)
% X - GPUtype
%
%
% DESCRIPTION
% Prints GPU type information. DISPLAY(X) is called for the ob-
% ject X when the semicolon is not used to terminate a statement.
%
% EXAMPLE
%
% A = GPUsingle(rand(10));
% display(A)
% A

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

if (GPUcompileMode==1)
  return;
end

gpuFLOAT   = 0;
gpuCFLOAT  = 1;
gpuDOUBLE  = 2;
gpuCDOUBLE = 3;
gpuINT32   = 4;
type = getType(p);
switch type
  case gpuCFLOAT
    single(p)
    disp('Single precision COMPLEX GPU type.');
  case  gpuFLOAT
    single(p)
    disp('Single precision REAL GPU type.');
  case gpuCDOUBLE
    double(p)
    disp('Double precision COMPLEX GPU type.');
  case  gpuDOUBLE
    double(p)
    disp('Double precision REAL GPU type.');
  case gpuINT32
    int32(p)
    disp('Int32 precision REAL GPU type.');
  otherwise 
    disp('Unrecognized GPU type.');
    
end

end
