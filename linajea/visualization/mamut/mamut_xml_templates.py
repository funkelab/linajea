# Template pre-spots (root and features).
# flake8: noqa
begin_template = '''<?xml version="1.0" encoding="UTF-8"?>
<TrackMate version="3.4.2">
  <Model spatialunits="pixels" timeunits="frames">
    <FeatureDeclarations>
      <SpotFeatures>
        <Feature feature="QUALITY" name="Quality" shortname="Quality" dimension="QUALITY" isint="false" />
        <Feature feature="POSITION_X" name="X" shortname="X" dimension="POSITION" isint="false" />
        <Feature feature="POSITION_Y" name="Y" shortname="Y" dimension="POSITION" isint="false" />
        <Feature feature="POSITION_Z" name="Z" shortname="Z" dimension="POSITION" isint="false" />
        <Feature feature="POSITION_T" name="T" shortname="T" dimension="TIME" isint="false" />
        <Feature feature="FRAME" name="Frame" shortname="Frame" dimension="NONE" isint="true" />
        <Feature feature="RADIUS" name="Radius" shortname="R" dimension="LENGTH" isint="false" />
        <Feature feature="VISIBILITY" name="Visibility" shortname="Visibility" dimension="NONE" isint="true" />
        <Feature feature="SOURCE_ID" name="Source ID" shortname="Source" dimension="NONE" isint="true" />
        <Feature feature="CELL_DIVISION_TIME" name="Cell division time" shortname="Cell div. time" dimension="TIME" isint="false" />
        <Feature feature="MANUAL_COLOR" name="Manual spot color" shortname="Spot color" dimension="NONE" isint="true" />
      </SpotFeatures>
      <EdgeFeatures>
        <Feature feature="SPOT_SOURCE_ID" name="Source spot ID" shortname="Source ID" dimension="NONE" isint="true" />
        <Feature feature="SPOT_TARGET_ID" name="Target spot ID" shortname="Target ID" dimension="NONE" isint="true" />
        <Feature feature="LINK_COST" name="Link cost" shortname="Cost" dimension="NONE" isint="false" />
        <Feature feature="VELOCITY" name="Velocity" shortname="V" dimension="VELOCITY" isint="false" />
        <Feature feature="DISPLACEMENT" name="Displacement" shortname="D" dimension="LENGTH" isint="false" />
        <Feature feature="MANUAL_COLOR" name="Manual edge color" shortname="Edge color" dimension="NONE" isint="true" />
        <Feature feature="TIME" name="Time" shortname="Time" dimension="NONE" isint="true" />
        <Feature feature="TISSUE_TYPE" name="Tissue id" shortname="T id" dimension="NONE" isint="true" />
      </EdgeFeatures>
      <TrackFeatures>
        <Feature feature="TRACK_INDEX" name="Track index" shortname="Index" dimension="NONE" isint="true" />
        <Feature feature="TRACK_ID" name="Track ID" shortname="ID" dimension="NONE" isint="true" />
        <Feature feature="TRACK_DURATION" name="Duration of track" shortname="Duration" dimension="TIME" isint="false" />
        <Feature feature="TRACK_START" name="Track start" shortname="T start" dimension="TIME" isint="false" />
        <Feature feature="TRACK_STOP" name="Track stop" shortname="T stop" dimension="TIME" isint="false" />
        <Feature feature="TRACK_DISPLACEMENT" name="Track displacement" shortname="Displacement" dimension="LENGTH" isint="false" />
        <Feature feature="NUMBER_SPOTS" name="Number of spots in track" shortname="N spots" dimension="NONE" isint="true" />
        <Feature feature="NUMBER_GAPS" name="Number of gaps" shortname="Gaps" dimension="NONE" isint="true" />
        <Feature feature="LONGEST_GAP" name="Longest gap" shortname="Longest gap" dimension="NONE" isint="true" />
        <Feature feature="NUMBER_SPLITS" name="Number of split events" shortname="Splits" dimension="NONE" isint="true" />
        <Feature feature="NUMBER_MERGES" name="Number of merge events" shortname="Merges" dimension="NONE" isint="true" />
        <Feature feature="NUMBER_COMPLEX" name="Complex points" shortname="Complex" dimension="NONE" isint="true" />
        <Feature feature="DIVISION_TIME_MEAN" name="Mean cell division time" shortname="Mean div. time" dimension="TIME" isint="false" />
        <Feature feature="DIVISION_TIME_STD" name="Std cell division time" shortname="Std div. time" dimension="TIME" isint="false" />
      </TrackFeatures>
    </FeatureDeclarations>\n'''

# <Feature feature="TISSUE_NAME" name="Tissue name" shortname="T name" dimension="NONE" isint="false" />
# <Feature feature="TISSUE_NAME" name="Tissue name" shortname="T name" dimension="NONE" isint="false" />
# <Feature feature="TISSUE_TYPE" name="Tissue id" shortname="T id" dimension="NONE" isint="true" />

# Templates for spots.
allspots_template =     '    <AllSpots nspots="{nspots}">\n'
inframe_template =      '     <SpotsInFrame frame="{frame}">\n'
spot_template =         '        <Spot ID="{id}" name="{name}" VISIBILITY="1" RADIUS="10.0" QUALITY="{quality}" SOURCE_ID="0" POSITION_T="{frame}.0" POSITION_X="{x}" POSITION_Y="{y}" FRAME="{frame}" POSITION_Z="{z}"/>\n'
inframe_end_template =  '     </SpotsInFrame>\n'
allspots_end_template = '    </AllSpots>\n'
inframe_empty_template = '     <SpotsInFrame frame="{frame}" />\n'

# Templates for tracks and edges.
alltracks_template =        '    <AllTracks>\n'
track_template =            '      <Track name="Track_{id}" TRACK_INDEX="{id}" TRACK_ID="{id}" TRACK_DURATION="{duration}.0" TRACK_START="{start}" TRACK_STOP="{stop}.0" TRACK_DISPLACEMENT="0" NUMBER_SPOTS="{nspots}" NUMBER_GAPS="0" LONGEST_GAP="0" NUMBER_SPLITS="0" NUMBER_MERGES="0" NUMBER_COMPLEX="0" DIVISION_TIME_MEAN="NaN" DIVISION_TIME_STD="NaN">\n'
edge_template =             '        <Edge SPOT_SOURCE_ID="{source_id}" SPOT_TARGET_ID="{target_id}" LINK_COST="-1.0" VELOCITY="0" DISPLACEMENT="0" TISSUE_TYPE="0" TIME="{time}" />\n'
# edge_template =             '        <Edge SPOT_SOURCE_ID="{source_id}" SPOT_TARGET_ID="{target_id}" LINK_COST="-1.0" VELOCITY="{velocity}" DISPLACEMENT="{displacement}" />\n'
track_end_template =        '      </Track>\n'
alltracks_end_template =    '    </AllTracks>\n'

# Templates for filtered tracks.
filteredtracks_start_template = '    <FilteredTracks>\n'
filteredtracks_template = '      <TrackID TRACK_ID="{t_id}" />\n'
filteredtracks_end_template = '    </FilteredTracks>\n'

      # <TrackID TRACK_ID="1" />\n      <TrackID TRACK_ID="2" />\n    </FilteredTracks>'

# Template for ending the XML file.
im_data_template = '<ImageData filename="{filename}" folder="{folder}" width="0" height="0" nslices="0" nframes="0" pixelwidth="1.0" pixelheight="1.0" voxeldepth="1.0" timeinterval="1.0" />'
end_template = '''
  </Model>
  <Settings>
    {image_data}
    <InitialSpotFilter feature="QUALITY" value="0.0" isabove="true" />
    <SpotFilterCollection />
    <TrackFilterCollection />
    <AnalyzerCollection>
      <SpotAnalyzers>
        <Analyzer key="Spot Source ID" />
        <Analyzer key="CELL_DIVISION_TIME_ON_SPOTS" />
        <Analyzer key="MANUAL_SPOT_COLOR_ANALYZER" />
      </SpotAnalyzers>
      <EdgeAnalyzers>
        <Analyzer key="Edge target" />
        <Analyzer key="Edge velocity" />
        <Analyzer key="MANUAL_EDGE_COLOR_ANALYZER" />
      </EdgeAnalyzers>
      <TrackAnalyzers>
        <Analyzer key="Track index" />
        <Analyzer key="Track duration" />
        <Analyzer key="Branching analyzer" />
        <Analyzer key="CELL_DIVISION_TIME_ANALYZER" />
      </TrackAnalyzers>
    </AnalyzerCollection>
  </Settings>
  <GUIState>
    <SetupAssignments>
      <ConverterSetups>
        <ConverterSetup>
          <id>0</id>
          <min>50.0</min>
          <max>1100.0</max>
          <color>-1</color>
          <groupId>0</groupId>
        </ConverterSetup>
      </ConverterSetups>
      <MinMaxGroups>
        <MinMaxGroup>
          <id>0</id>
          <fullRangeMin>0.0</fullRangeMin>
          <fullRangeMax>65535.0</fullRangeMax>
          <rangeMin>0.0</rangeMin>
          <rangeMax>65535.0</rangeMax>
          <currentMin>50.0</currentMin>
          <currentMax>1100.0</currentMax>
        </MinMaxGroup>
      </MinMaxGroups>
    </SetupAssignments>
    <Bookmarks />
  </GUIState>
</TrackMate>'''

    #<ImageData filename="{filename}.xml" folder="/media/nelas/Actinotroch/4d" width="1376" height="1040" nslices="{nslices}" nframes="{nframes}" pixelwidth="1.0" pixelheight="1.0" voxeldepth="1.0" timeinterval="1.0" />
