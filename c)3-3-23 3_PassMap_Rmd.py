#PassMap_Rmd of 3rd March,2023

#Function to draw the pitch
import matplotlib.pyplot as plt
import numpy as np

#Size of the pitch
pitchLengthX=120
pitchWidthY=80

match_id_required = 3032310
home_team_required ="Real Madrid"
away_team_required ="Barcelona"

# Load in the data
file_name=str(match_id_required)+'.json'

#Load in all match events 
import json
with open('dataset/data/events/'+file_name) as data_file:
    #print (mypath+'events/'+file)
    data = json.load(data_file)

#get the nested structure into a dataframe 
#store the dataframe in a dictionary with the match id as key (remove '.json' from string)
from pandas import json_normalize
df = json_normalize(data, sep = "_").assign(match_id = file_name[:-5])

#Find the passes
passes = df.loc[df['type_name'] == 'Pass'].set_index('id')

#Draw the pitch
from mplsoccer.pitch import Pitch
pitch=Pitch(pitch_color='grass', line_color='white', stripe=True)
fig,ax=pitch.draw()

for i,thepass in passes.iterrows():
    if thepass['team_name']==home_team_required:
        x=thepass['location'][0]
        y=thepass['location'][1]
        passCircle=plt.Circle((x,pitchWidthY-y),2,color="#6495ed")      
        passCircle.set_alpha(.2)   
        ax.add_patch(passCircle)
        dx=thepass['pass_end_location'][0]-x
        dy=thepass['pass_end_location'][1]-y

        passArrow=plt.Arrow(x,pitchWidthY-y,dx,-dy,width=0.6,color="#6495ed")
        ax.add_patch(passArrow)

fig.set_size_inches(10, 7)
fig.savefig('c)Output/3-PassMap_Rmd.pdf', dpi=100) 
plt.show()