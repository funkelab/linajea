%transforms Imaris spot format (vertices+pairwise edges) into ordered
%subtracts

function [tracklets connTrack]=ImarisSpotsDataToTracklets(vert,tIdx,aEdges)



%find candidates to start tracklets: basically nodes which do not appear
%twice in the edges list

[p u]=hist(double(aEdges(:)),[min(double(aEdges(:))): max(double(aEdges(:)))]);
pos=find(p~=2);

%we will store the starting point of each subtrack in seeds
seedsT=zeros(1000,1);
countT=0;
for kk=1:length(pos)
    p_kk=p(pos(kk));
    u_kk=u(pos(kk));
    if(p_kk==1)%end or beginning of a track
        %check if it is the beginning of the track (otherwise we need to
        %disregard)
        isBeginning=false;
        pp=find(aEdges(:,1)==u_kk);
        if(isempty(pp))
            pp=find(aEdges(:,2)==u_kk);
            if(tIdx(aEdges(pp,1))>tIdx(aEdges(pp,2)))
                isBeginning=true;
            end
        else
            if(tIdx(aEdges(pp,1))<tIdx(aEdges(pp,2)))
                isBeginning=true;
            end
        end
        
        %start a new track if is beginning is true
        if(isBeginning==true)
           countT=countT+1; 
           seedsT(countT)=u_kk;           
        end
        
    elseif(p_kk==3)%cell division
        pp=find(aEdges(:,1)==u_kk);
        for ii=1:length(pp)
            if(tIdx(aEdges(pp(ii),1))<tIdx(aEdges(pp(ii),2)))%daughter cell
                countT=countT+1;
                seedsT(countT)=aEdges(pp(ii),2);
            end
        end
        pp=find(aEdges(:,2)==u_kk);
        for ii=1:length(pp)
            if(tIdx(aEdges(pp(ii),2))<tIdx(aEdges(pp(ii),1)))%daughter cell
                countT=countT+1;
                seedsT(countT)=aEdges(pp(ii),1);
            end
        end
    else
       error 'Unexpected number of edges for a tree-like structure' 
    end
end

if(length(seedsT)>countT)
    seedsT(countT+1:end)=[];
end

%find each track starting from seed
tracklets=cell(countT,1);


auxT=zeros(500,5);%[t x y z id]
for kk=1:countT
    
    count=0;
    queue=seedsT(kk);
    
    while(isempty(queue)==false)
       parentId=queue(1);
       queue(1)=[];
       if(sum(parentId==seedsT)>0 && count>0)
           break;% we have reached another subtrack
       end
       
       %add point to tracklet
       count=count+1;
       auxT(count,:)=[double(tIdx(parentId)) double(vert(parentId,:)) parentId];
       
       %find daughter
       pp=find(aEdges(:,1)==parentId);
        for ii=1:length(pp)
            if(tIdx(aEdges(pp(ii),1))<tIdx(aEdges(pp(ii),2)))%daughter cell
                queue=[queue aEdges(pp(ii),2)];
            end
        end
        pp=find(aEdges(:,2)==parentId);
        for ii=1:length(pp)
            if(tIdx(aEdges(pp(ii),2))<tIdx(aEdges(pp(ii),1)))%daughter cell
                queue=[queue aEdges(pp(ii),1)];
            end
        end
       
    end
    
    %addd tracklet to cell array
    tracklets{kk}=auxT(1:count,:);
end

%find connected tracklets
countC=0;
connTrack=zeros(2*countT,3);

for kk=1:countT
    parentId=tracklets{kk}(end,5);
    
    pp1=find(aEdges(:,1)==parentId);
    pp2=find(aEdges(:,2)==parentId);
    aux=[aEdges(pp1,2); aEdges(pp2,1)];%candidates to be daughter
    
    daughter=intersect(aux,seedsT);
    if(isempty(daughter))%nothing to do
        continue;
    end
    
    countC=countC+1;
    if(length(daughter)==1)
        connTrack(countC,:)=[kk find(daughter==seedsT) -1];
    elseif(length(daughter)==2)
        connTrack(countC,:)=[kk find(daughter(1)==seedsT) find(daughter(2)==seedsT)];
    else
        error 'There should not be more than 2 daughters'
    end
end

if(size(connTrack,1)>countC)
    connTrack(countC+1:end,:)=[];
end






















