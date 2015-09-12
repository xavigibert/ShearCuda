function stopMemoryCheck

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


if (GPUtest.checkPointers==1)
  GPUmem; % clear memory cache
  if exist(fullfile('.','GPUtype.dat'),'file') && exist(fullfile('.','GPUtypedelete.dat'),'file')

    [m,f] = compareAllocFree(fullfile('.','GPUtype.dat'),fullfile('.','GPUtypedelete.dat'));

    if (isempty(find(m)))
    else
      error('checkMemory');
    end

    if (isempty(find(f)))
    else
      error('checkMemory');
    end
  else
    warning('Cannot find file');
  end

  if exist(fullfile('.','gpumalloc.dat'),'file') && exist(fullfile('.','gpufree.dat'),'file')
    [m,f] = compareAllocFree(fullfile('.','gpumalloc.dat'),fullfile('.','gpufree.dat'));

    if (isempty(find(m)))
    else
      error('checkMemory');
    end

    if (isempty(find(f)))
    else
      error('checkMemory');
    end
  else
    warning('Cannot find file');
  end

  if exist(fullfile('.','malloc.dat'),'file') && exist(fullfile('.','free.dat'),'file')

    [m,f] = compareAllocFree(fullfile('.','malloc.dat'),fullfile('.','free.dat'));

    if (isempty(find(m)))
    else
      error('checkMemory');
    end

    if (isempty(find(f)))
    else
      error('checkMemory');
    end

  else
    warning('Cannot find file');
  end

end





function a = myload(file)
fid=fopen(file);
a =[];
while 1
  tline = fgetl(fid);
  if ~ischar(tline), break, end
  a(end+1) = hex2num(strrep(tline,'0x',''));
end
fclose(fid);

function [malloc,free]= compareAllocFree(m,f)

malloc = myload(m);
free = myload(f);
for i=1:length(malloc)
  j = find(free==malloc(i));
  if isempty(j)
  else
    malloc(i) = 0;
    free(j(1)) = 0;
  end
end


