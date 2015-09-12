function GPUfor (varargin)
%GPUfor Starts GPU for-loop (Only in compilation mode)
%

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

% Create the iterator in caller workspace
it = (cell2mat(varargin));
try
  evalin('caller', [it ';']);
catch
  GPUcompileAbort
  err = lasterror;
  txt = sprintf('%s\n%s\n',err.message, ['Unable to parse iterator ''' it '''. (ERROR code GPUfor.1)']);
  error(txt);
end

% get variable name
itname1 = strtok(it,'=');
if (length(itname1)==length(it))
  GPUcompileAbort
  error(['Unable to parse iterator ''' it '''. (ERROR code GPUfor.2)'] );
end
itname = strtrim(itname1);


% i and j are not allowed names for iterators
if (strcmp(itname,'i') || strcmp(itname,'j'))
  GPUcompileAbort
  txt = 'Iterator name cannot be ''i'' or ''j''. (ERROR code GPUfor.3)';
  error(txt);
end

evalin('caller', ['GPUcompileForStart (' itname ')']);

% set iterator to the first value
evalin('caller',[itname '=' itname '(1);']);


end