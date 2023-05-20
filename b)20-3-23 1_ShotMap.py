#ShotMap of 20th March,2023
#Make a shot map

#Function to draw the pitch
import matplotlib.pyplot as plt
import numpy as np

#Size of the pitch
pitchLengthX=120
pitchWidthY=80

match_id_required = 20032321
home_team_required ="Barcelona"
away_team_required ="Real Madrid"

# Load in the data
file_name=str(match_id_required)+'.json'

#Load in all match events 
import json
with open('dataset/data/events/'+file_name) as data_file:
    #print (mypath+'events/'+file)
    data = json.load(data_file)

#get the nested structure into a dataframe 
#store the dataframe in a dictionary with the match id as key (remove '.json' from string)
from pandas.io.json import json_normalize
df = json_normalize(data, sep = "_").assign(match_id = file_name[:-5])

#A dataframe of shots
shots = df.loc[df['type_name'] == 'Shot'].set_index('id')
    
#Draw the pitch
from mplsoccer.pitch import Pitch
pitch=Pitch(pitch_color='grass', line_color='white', stripe=True)
fig,ax=pitch.draw()

#Plot the shots
for i,shot in shots.iterrows():
    x=shot['location'][0]
    y=shot['location'][1]
    
    goal=shot['shot_outcome_name']=='Goal'
    team_name=shot['team_name']
    
    circleSize=2

    if (team_name==home_team_required):
        if goal:
            shotCircle=plt.Circle((x,pitchWidthY-y),circleSize,color="#be0032")
        else:
            shotCircle=plt.Circle((x,pitchWidthY-y),circleSize,color="#be0032")     
            shotCircle.set_alpha(.4)
    elif (team_name==away_team_required):
        if goal:
            shotCircle=plt.Circle((pitchLengthX-x,y),circleSize,color="#6495ed") 
        else:
            shotCircle=plt.Circle((pitchLengthX-x,y),circleSize,color="#6495ed")      
            shotCircle.set_alpha(.6)
    ax.add_patch(shotCircle)
    
    
plt.text(5,75,away_team_required + ' shots') 
plt.text(80,75,home_team_required + ' shots') 
     
fig.set_size_inches(10, 7)
fig.savefig('b)Output/1-ShotMap.pdf', dpi=100) 
plt.show()