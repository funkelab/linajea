function [vImarisApplicationVec]=openImarisConnectionAll()
%---------------------------------------------------------
%-----basic interface to setup the pipe betwen Matlab and Imaris----------
% connect to Imaris interface: Imaris should be running!!!
javaaddpath C:\Program' Files'\Bitplane\Imaris' x64 7.4.0'\XT\Matlab\ImarisLib.jar
vImarisLib = ImarisLib;
vServer = vImarisLib.GetServer;

N=vServer.GetNumberOfObjects;
vImarisApplicationVec=cell(N,1);

for kk=1:N
    vObjectId=vServer.GetObjectID(kk-1);
    vImarisApplicationVec{kk} = vImarisLib.GetApplication(vObjectId);
end