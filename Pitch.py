# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 10:55:29 2023

@author: haree
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Arc

def createPitch():
    
    #Create figure
    fig=plt.figure()
    fig.patch.set_facecolor('#006400')
    ax=fig.add_subplot(1,1,1)

    #Pitch Outline & Centre Line
    plt.plot([0,0],[0,90], color="white")
    plt.plot([0,130],[90,90], color="white")
    plt.plot([130,130],[90,0], color="white")
    plt.plot([130,0],[0,0], color="white")
    plt.plot([65,65],[0,90], color="white")
    
    #Left Penalty Area
    plt.plot([16.5,16.5],[65,25],color="white")
    plt.plot([0,16.5],[65,65],color="white")
    plt.plot([16.5,0],[25,25],color="white")
    
    #Right Penalty Area
    plt.plot([130,113.5],[65,65],color="white")
    plt.plot([113.5,113.5],[65,25],color="white")
    plt.plot([113.5,130],[25,25],color="white")
    
    #Left 6-yard Box
    plt.plot([0,5.5],[54,54],color="white")
    plt.plot([5.5,5.5],[54,36],color="white")
    plt.plot([5.5,0.5],[36,36],color="white")
    
    #Right 6-yard Box
    plt.plot([130,124.5],[54,54],color="white")
    plt.plot([124.5,124.5],[54,36],color="white")
    plt.plot([124.5,130],[36,36],color="white")
    
    #Prepare Circles
    centreCircle = plt.Circle((65,45),9.15,color="white",fill=False)
    centreSpot = plt.Circle((65,45),0.8,color="white")
    leftPenSpot = plt.Circle((11,45),0.8,color="white")
    rightPenSpot = plt.Circle((119,45),0.8,color="white")
    
    #Draw Circles
    ax.add_patch(centreCircle)
    ax.add_patch(centreSpot)
    ax.add_patch(leftPenSpot)
    ax.add_patch(rightPenSpot)
    
    #Prepare Arcs
    leftArc = Arc((11,45),height=18.3,width=18.3,angle=0,theta1=310,theta2=50,color="white")
    rightArc = Arc((119,45),height=18.3,width=18.3,angle=0,theta1=130,theta2=230,color="white")

    #Draw Arcs
    ax.add_patch(leftArc)
    ax.add_patch(rightArc)
    
    #Tidy Axes
    plt.axis('off')
    
    #Display Pitch
    plt.show()
    
createPitch()