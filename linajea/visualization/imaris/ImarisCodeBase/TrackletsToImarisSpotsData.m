function [vert,tIdx,aEdges]=TrackletsToImarisSpotsData(tracklets,connTrack)


numT=length(tracklets);
%calculate total number of spots
numSpots=0;
for kk=1:numT
    numSpots=numSpots+size(tracklets{kk},1);
end

%save vert, tIdx and edges
vert=zeros(numSpots,3);
tIdx=zeros(numSpots,1);
aEdges=zeros(2*numSpots,2);
numE=0;

offset=ones(numT+1,1);%offset for each tracklet within vert. +1 to handle last tracklet smoothly in the code without if statements.
for kk=1:numT
    N=size(tracklets{kk},1);
    offset(kk+1)=offset(kk)+N;

    vert(offset(kk):offset(kk+1)-1,:)=tracklets{kk}(:,2:4);
    tIdx(offset(kk):offset(kk+1)-1)=tracklets{kk}(:,1);
    aEdges(numE+1:numE+N-1,:)=[[offset(kk):offset(kk+1)-2]' 1+[offset(kk):offset(kk+1)-2]'];
    numE=numE+N-1;
end

%add final edges between different tracklets
for kk=1:size(connTrack,1)
    parent=connTrack(kk,1);
    for ii=1:2
        daughter=connTrack(kk,1+ii);
        if(daughter<=0) continue;end;
        
        numE=numE+1;
        aEdges(numE,:)=sort([offset(parent+1)-1 offset(daughter)],'ascend');
    end    
end

if(size(aEdges,1)>numE)
    aEdges(numE+1:end,:)=[];
end



%transform each array to its correct format for Imaris
vert=single(vert);
tIdx=int32(tIdx);
aEdges=int32(aEdges);