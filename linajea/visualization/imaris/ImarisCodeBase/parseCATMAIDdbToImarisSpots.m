
% INPUT:
% 
%
%trackingM:               array Nx 10 retrived from tracking information in CATMAID databse
%information from CATMAID database
%stackRes:                array 3x1 indicating resolution (needed to parse xyz from CATMAID (in um) to pixels    
%sourceVolumeDimensions:  array 3x1 indicating dimensions of image stack.
%Use [] if you did not scale the volume in Imaris
%targetVolumeDimensions:  array 3x1 to use "fake" imaris volume. Use [] if
%the volume at Imaris has the same dimensions as sourceVolumeDimensions


%trackingM matrix contains teh following columns
%id, type, x, y, z, radius, parent_id, time, confidence, skeleton_id
function parseCATMAIDdbToImarisSpots(trackingM, stackRes,sourceVolumeDimensions, targetVolumeDimensions)

%parse vertices
disp 'calculating vertices ...'
tIdx = trackingM(:,8);
vert = trackingM(:,3:5);
for kk =1 :3
   vert(:,kk) = vert(:, kk ) / stackRes(:,kk);
end

%scale vertices
if( isempty(sourceVolumeDimensions) == false )
    vert = single(vert);
    for kk = 1:size(vert,2)
        vert(:,kk) = vert(:,kk) * (targetVolumeDimensions(kk) / sourceVolumeDimensions(kk));
    end
end


%calculating edges
disp 'calculating edges ...'
pos = find( trackingM(:,7) >= 0);%all elements with an edge
nodeIdMap = containers.Map(trackingM(:,1),[1:size(trackingM,1)]);
aEdges = [pos pos];%to reserve memory
for kk = 1: length(pos)%container.Map needs cell array input to vectorize output
   aEdges(kk,2) = nodeIdMap(trackingM(pos(kk),7)); 
end


%update spots to Imaris
disp 'Updating spots in Imaris...'
tIdx=int32(tIdx);
vert=single(vert);
radii=ones(size(tIdx),'single');
aEdges=int32(aEdges);
setSpots(vert,tIdx,radii,aEdges,[1 1 1],[0 0 0 0]);%we assume scale is [1 1 1] since results have been conducted in teh same dataset