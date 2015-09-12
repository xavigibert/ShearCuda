function r = double(g)
% double - Converts a GPU variable into a Matlab single precision
% variable
%
% SYNTAX
%
% R = double(X)
% X - GPUtype
% R - Matlab variable
%
%
% DESCRIPTION
% B = DOUBLE(A) returns the contents of the GPU variable A into a
% double precision Matlab array.
%
% Complex type supported since version 0.1
% Real type supported since version 0.1
%
% EXAMPLE
%
% A = GPUdouble(rand(100))
% Ah = double(A);

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


if (isempty(g))
  r = [];
  return;
end
r=double(GPUtypeToMxNumericArray(g));

end

