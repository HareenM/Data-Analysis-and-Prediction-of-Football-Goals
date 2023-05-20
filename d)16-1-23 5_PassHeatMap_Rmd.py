#Pass Heat Map of Real Madrid

#Function to draw the pitch
import matplotlib.pyplot as plt
import numpy as np
import json
from pandas import json_normalize
from FCPython import createPitch

#Size of the pitch
pitchLengthX=120
pitchWidthY=80

team_required ="Real Madrid"

#Get the list of matches
competition_id=11
#Load the list of matches for this competition
with open('dataset/data/matches/'+str(competition_id)+'/0.json', encoding="utf8") as f:
    matches = json.load(f)

#Find the matches they played
match_id_required=[]
for match in matches:
    home_team_name=match['home_team']['home_team_name']
    away_team_name=match['away_team']['away_team_name']
    if (home_team_name==team_required) or (away_team_name==team_required):
        match_id_required.append(match['match_id'])
        
#Find the passes for each match
for ic,match_id in enumerate(match_id_required):
    
    #Load in all match events 

    file_name=str(match_id)+'.json'
    with open('dataset/data/events/'+file_name, encoding="utf8") as data_file:
        data = json.load(data_file)
    df = json_normalize(data, sep = "_").assign(match_id = file_name[:-5])
    team_actions = (df['team_name']==team_required)
    df = df[team_actions]
    
    #A dataframe of passes
    passes_match = df.loc[df['type_name'] == 'Pass'].set_index('id')
    #A dataframe of shots
    shots_match = df.loc[df['type_name'] == 'Shot'].set_index('id')
    
    #Find shot times in seconds
    #This should be adjusted to account for overlapping halves of the match.
    shot_times = shots_match['minute']*60+shots_match['second']
    shot_window = 15  
    shot_start = shot_times - shot_window
    pass_times = passes_match['minute']*60+passes_match['second']
    
    #Check with passes are whitin [shot_window] seconds of a shot
    def in_range(pass_time,start,finish):
        return (True in ((start < pass_time) & (pass_time < finish)).unique())

    pass_to_shot = pass_times.apply(lambda x: in_range(x,shot_start,shot_times))
    
    #Exclude corners
    iscorner = passes_match['pass_type_name']=='Corner'
    
    danger_passes=passes_match[np.logical_and(pass_to_shot,np.logical_not(iscorner))]
    
    if ic==0:
        passes =  danger_passes
    else:
        passes = passes.append(danger_passes)

    
    
    print('Match: ' + str(match_id) + '. Number of danger passes is: ' + str(len(danger_passes)))


#Set number of matches
number_of_matches=ic+1

#Size of the pitch in yards (!!!)
pitchLengthX=120
pitchWidthY=80

#Draw the pitch
from mplsoccer.pitch import Pitch
pitch=Pitch(pitch_color='grass', line_color='white', stripe=True)
fig,ax=pitch.draw()

#Plot the passes
for i,thepass in passes.iterrows():
    x=thepass['location'][0]
    y=pitchWidthY-thepass['location'][1]
    passCircle=plt.Circle((x,y),1,color="#6495ed")      
    passCircle.set_alpha(.5)   
    ax.add_patch(passCircle)

ax.set_title('Danger passes by ' + team_required)
fig.set_size_inches(10, 7)
fig.savefig('d)Output/5-PassHeatMap_Rmd.pdf', dpi=100) 
plt.show()

#Make x,y positions
x=[]
y=[]
for i,apass in passes.iterrows():
    x.append(apass['location'][0])
    y.append(pitchWidthY-apass['location'][1])

#Make a histogram of passes
H_Pass=np.histogram2d(y, x,bins=5,range=[[0, pitchWidthY],[0, pitchLengthX]])
