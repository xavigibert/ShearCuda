function a = getAbsolutePath(relpath)

curdir = pwd;
cd(relpath);
a = pwd;
cd(curdir);
end