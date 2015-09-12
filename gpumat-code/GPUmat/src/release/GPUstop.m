function y = GPUstop()
%GPUstop Stops the GPU environment


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

% next line required for 2009a
warning off MATLAB:structOnObject


if (~mislocked('GPUmat'))
  disp('GPU not started');
  return
end

% Clear variables
S = evalin('base','whos');
for i=1:length(S)
  if (strcmp(S(i).class,'GPUsingle')||strcmp(S(i).class,'GPUdouble'))
    evalin('base',['clear ' S(i).name]);
  end
  
end

% Delete GPUmanager
GPUmanagerCreate(0,0,0,0,1,0,0);
munlock('GPUmanager');
clear GPUmanager

% Need to unlock some other functions
clear GPUtypeDelete

munlock('GPUmat');
clear GPUmat

end
