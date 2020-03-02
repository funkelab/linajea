function updateBatchToDatabase(xMin,yMin,zMin,iniFrame,vert,tIdx,aEdges,experimentName,datasetName,proofReadSpots)

userDB='Fernando'
ppssww='Pqqld5';


%offset all the anootations

%{
1.-TIF datasets were read in Matlab using imread. 
2.-Each dataset was cropped using:
        im1(yMin:yMax,xMin:xMax,zMin:zMax)
3.-Offset=[xMin-1 yMin-1 zMin-1 iniFrame]  %It seems Imaris also permutes x,y with respect to Matlab as V3D does
%}


offset=[xMin-1 yMin-1 zMin-1 iniFrame];


%apply offset
vert=vert+repmat(offset(1:3),[size(vert,1) 1]);%apply translation offset
tIdx=tIdx+offset(4);
proofReadSpots=proofReadSpots+repmat(offset,[size(proofReadSpots,1) 1]);

%-----------------------------------------------------
%open database connection if it is not open already
qq=mym('status');
if(qq>0)
    mym('open','localhost',userDB,ppssww);
end
mym('use tracking_cells');%set the appropriate database

%------------------------------------------------------------------------------------
%check if dataset exists in database. Otherwise add it

sqlCmd=['SELECT dataset_id FROM datasets WHERE name = ''' datasetName  ''' '];
sqlVal=mym(sqlCmd);

if(isempty(sqlVal.dataset_id))
   sqlCmd=['INSERT INTO datasets (name,comments) VALUES (''' datasetName ''',''none'')'];
   mym(sqlCmd);
   
   qq=mym('SELECT LAST_INSERT_ID()');
   datasetId=getfield(qq,'LAST_INSERT_ID()');
elseif(length(sqlVal.dataset_id)>1)
    error 'Dataset name has more than one entry in database'
else
    datasetId=sqlVal.dataset_id;
end



%------------------------------------------------------------------------------
%decide parent and children for each vertex
N=length(tIdx);
parentId=-ones(N,1);%-1 indicate no parent or no children (it will be null value in the database)
ch1=parentId;
ch2=parentId;

for kk=1:size(aEdges,1)
   e1=aEdges(kk,1);
   e2=aEdges(kk,2);
   
   if(tIdx(e1)>tIdx(e2))%swap e1 and e2
       e2=aEdges(kk,1);
       e1=aEdges(kk,2);
   end
   
   %we assume e1 is in time t and e2 is in time t2
   if(parentId(e2)>0)
       error 'Parent has already been assigned'
   end
   parentId(e2)=e1;
   if(ch1(e1)<0)
       ch1(e1)=e2;
   else
       if(ch2(e1)<0)
           ch2(e1)=e2;
       else
           error 'Spot already has two children'
       end
   end
end


%create proof read array
proofReadCheck=zeros(N,1);
[idx dist]=knnsearch([vert tIdx],proofReadSpots);
if(max(dist)>1e-3)
    error 'We can not find a match for some of the proof read spots'
end
proofReadCheck(idx)=1;


%check if experiment already exists in database. If it does, delete all the
%elements before uploading news
sqlCmd=['DELETE FROM centroids WHERE comments = ''' experimentName  ''' '];
mym(sqlCmd);




%update database
cellIdMap=zeros(N,1);%to map cell_id between Matlab and adtabase
for kk=1:N
    
    sqlCmd=['INSERT INTO centroids (x,y,z,time_point,dataset_id,proof_read,comments) VALUES (' ...
            num2str(vert(kk,1)) ',' num2str(vert(kk,2)) ',' num2str(vert(kk,3)) ',' num2str(tIdx(kk)) ',' ...
            num2str(datasetId) ',' num2str(proofReadCheck(kk)) ', ''' experimentName  ''')'];
    mym(sqlCmd);
    
    %obtain inserted id
    qq=mym('SELECT LAST_INSERT_ID()');
    cellIdMap(kk)=getfield(qq,'LAST_INSERT_ID()');
end

%translate parentID and childrenID to database id (it should be a simple
%offset) and update record
parentId(parentId>0)=cellIdMap(parentId(parentId>0));
ch1(ch1>0)=cellIdMap(ch1(ch1>0));
ch2(ch2>0)=cellIdMap(ch2(ch2>0));

for kk=1:N
   %update entries given that we have the cellId value
   if(parentId(kk)<=0)
       pp='null';
   else
       pp=num2str(parentId(kk));
   end
   
   if(ch1(kk)<=0)
       c1='null';
   else
       c1=num2str(ch1(kk));
   end
   
   if(ch2(kk)<=0)
       c2='null';
   else
       c2=num2str(ch2(kk));
   end
   
   sqlCmd=['UPDATE centroids SET parent_id=' pp ',child1_id=' c1 ',child2_id=' c2 ' WHERE cell_id=' num2str(cellIdMap(kk))];
   mym(sqlCmd); 
end







