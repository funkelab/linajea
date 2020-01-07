%warning: aEdges should be in Matlab indexing; so 1 is the minimum value
function setSpots(vert,tIdx,radii,aEdges,scale,offset)

%offset=[0 0 0 0]%[X y Z t] offset in case we have cropped the volume
%scale=[1 1 5]

%connect to both Imaris sessions
[vImarisApplication]=openImarisConnection;


aSpots = vImarisApplication.GetFactory.ToSpots(vImarisApplication.GetSurpassSelection);
vImarisDataSet = vImarisApplication.GetDataSet;
vDataMin = [vImarisDataSet.GetExtendMinX, vImarisDataSet.GetExtendMinY, vImarisDataSet.GetExtendMinZ];
vDataMax = [vImarisDataSet.GetExtendMaxX, vImarisDataSet.GetExtendMaxY, vImarisDataSet.GetExtendMaxZ];
vDataSize = [vImarisDataSet.GetSizeX, vImarisDataSet.GetSizeY, vImarisDataSet.GetSizeZ];


auxSc=vDataSize./(vDataMax-vDataMin);




%apply offset
tIdx=tIdx+offset(4);
vert=vert+repmat(offset(1:3),[size(vert,1) 1]);%apply translation offset

%apply scaling
vert=vert./repmat(scale,[size(vert,1) 1]);


%convert points to Imaris frame units
vert=vert./repmat(auxSc,[size(vert,1) 1]);


%upload points to volume
aSpots.Set(vert,tIdx,radii);%we assume radius=1
aSpots.SetTrackEdges(aEdges-1);%C-indexing
