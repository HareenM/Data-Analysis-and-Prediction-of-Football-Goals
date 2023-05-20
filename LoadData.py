# -*- coding: utf-8 -*-
"""
Created on Mon May  1 07:37:36 2023

@author: haree
"""

#Importing the library required for loading json files
import json

#Opening the json files
with open('open-data-master/data/competitions.json', encoding='utf-8') as f:
    competitions=json.load(f)
    
#LaLiga has competition id 11
competition_id=11

#Loading the list of matches for this competition
with open('open-data-master/data/matches/'+str(competition_id)+'/27.json', encoding='utf-8') as f:
    matches=json.load(f)

#Print all matches    
for match in matches:
    home_team_name=match['home_team']['home_team_name']
    away_team_name=match['away_team']['away_team_name']
    home_score=match['home_score']
    away_score=match['away_score']
    describe_text = 'The match between ' + home_team_name + ' and ' + away_team_name
    result_text = ' finished ' + str(home_score) +  ' : ' + str(away_score)
    print(describe_text + result_text)
