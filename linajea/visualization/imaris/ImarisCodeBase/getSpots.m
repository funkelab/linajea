function [vImarisApplication vert tIdx radii aEdges trackId]=getSpots(offset,vImarisApplication)

%offset=[259 190 49  0]%[X y Z t] offset in case we have cropped the volume


%connect to both Imaris sessions
if(isempty(vImarisApplication))
    [vImarisApplication]=openImarisConnection;
end

aSpots = vImarisApplication.GetFactory.ToSpots(vImarisApplication.GetSurpassSelection);
vImarisDataSet = vImarisApplication.GetDataSet;
vDataMin = [vImarisDataSet.GetExtendMinX, vImarisDataSet.GetExtendMinY, vImarisDataSet.GetExtendMinZ];
vDataMax = [vImarisDataSet.GetExtendMaxX, vImarisDataSet.GetExtendMaxY, vImarisDataSet.GetExtendMaxZ];
vDataSize = [vImarisDataSet.GetSizeX, vImarisDataSet.GetSizeY, vImarisDataSet.GetSizeZ];


auxSc=vDataSize./(vDataMax-vDataMin);


%get XYZ, time and radius
tIdx=aSpots.GetIndicesT();
vert=aSpots.GetPositionsXYZ();
radii=aSpots.GetRadii();

%get track edges and id
disp 'WARNING: we added already +1 to aEdges to set it as Matlab-indexing'
aEdges = aSpots.GetTrackEdges()+1;%to set it in Matlab coordinates instead of C-indexing
trackId=aSpots.GetTrackIds;

%apply offset
tIdx=tIdx+offset(4);
vert=vert.*repmat(auxSc,[size(vert,1) 1]);%center og mass in returned in metric not in pixels
vert=vert+repmat(offset(1:3),[size(vert,1) 1]);%apply translation offset
