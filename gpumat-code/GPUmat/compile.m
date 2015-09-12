function compile

global compilepar

%make cpp lib cuda clean

curdir = pwd;

%% First compile libraries
cd(fullfile('src','lib'));
make cpp lib install

cd(curdir)

cd(fullfile('src','cuda'));
make cpp lib install

%% now build the rest
cd(curdir)

make cpp cuda install




end



