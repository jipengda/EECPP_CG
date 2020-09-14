#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 23:26:27 2019

@author: jipeng
"""
import math
import numpy as np

def distance(firstNode, secondNode,coord_x,coord_y):
    j=secondNode
    i=firstNode
    distanceValue=0
    distanceValue=np.hypot(coord_x[i]-coord_x[j],coord_y[i]-coord_y[j])
    return distanceValue

def angle(second_to_lastNode, lastNode, newNode, coord_x, coord_y):
    radians_to_degrees = 180/(math.pi)
    theta_radians=0
    theta_degrees=0
    o=second_to_lastNode
    p=lastNode
    q=newNode
    distance_o_p=distance(o,p,coord_x,coord_y)
    distance_p_q=distance(p,q,coord_x,coord_y)
    distance_o_q=distance(o,q,coord_x,coord_y)
    theta_radians=math.pi-np.arccos(round((distance_o_p**2+distance_p_q**2-distance_o_q**2)/(2*distance_o_p*distance_p_q),2))
    theta_degrees=radians_to_degrees*theta_radians
    return theta_degrees

def totalCost_calculation_by_set(optimalSet, coord_x, coord_y, D):
    distance_lambda = 0.1164
    turn_gamma = 0.0173
    turn_gamma = 0.015 # so turning cost is 0
    totalCost = 0
    lastNode=0
    newNode=optimalSet[1]
    turnCost=0
    distanceCost= distance_lambda * D[lastNode][newNode]
    totalCost = turnCost + distanceCost
    length=len(optimalSet)
    for i in range(2, length):
        newNode = optimalSet[i]
        lastNode = optimalSet[i-1]
        second_to_lastNode = optimalSet[i-2]
        go = lastNode
        to = newNode
        turnCost = turn_gamma * angle(second_to_lastNode, go, to, coord_x, coord_y)
        distanceCost = distance_lambda * D[go][to]
        totalCost=totalCost+turnCost + distanceCost
    return totalCost