# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 15:47:38 2018

@author: dugj2403
"""
import os
import numpy as np
from copy import deepcopy
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def find_3D_tracks(path,prefix):
    """Recursively look in subdirectories for 3D track files.
    
    Args:
    path   (str): Path to folder to look in.
    param2 (str): User selected prefix used to identify 3D tracks.
    
    Returns:
    bool: list of file names containing prefix in path.
    """
    tracks=[]
    result = [y for x in os.walk(path) for y in glob(os.path.join(x[0], '*.csv'))]
    for i in result:
        if prefix in i:
            tracks.append(i)
    return tracks

def convert_to_dataframes(csvList):
    """Convert list of 3D tracks into pandas dataframes.
    
    Args:
    csvList  (str): list containing 3D tracks to be converted to dataframes
    
    Returns:
    bool: list of pandas dataframes containing 3D tracks.
    """
    dataframes=[]
    for file in csvList:
        dat = pd.read_csv(file, sep=',', engine='python') 
    
        dataframes.append(dat)
    return dataframes


def plot_2D_lines(df,columns):
    """Plot multiple columms of a pandas dataframe on a line graph.
    
    Args:
    df      (dataframe): pandas dataframe to plot, must contain x,y and z for now
    columns      (list): list containing xmin,xmax,ymin,ymax,zmin,zmax
    
    Returns:
    bool: list of pandas dataframes containing 3D tracks.
    """

    if type(columns) is list:
        
        ax = df.plot(y=columns[0],use_index=True)
        for i in range(len(columns)-1):
            df.plot(y=columns[i+1],ax=ax,use_index=True)

def plot_3D_points(df,**keyword_parameters):
    """Convert list of 3D tracks into pandas dataframes.
    
    Args:
    df      (dataframe): pandas dataframe to plot
    limits       (list): list containing xmin,xmax,ymin,ymax,zmin,zmax
    
    Returns:
    bool: list of pandas dataframes containing 3D tracks.
    """

    ax = plt.figure().gca(projection='3d')
    ax.scatter(df.x, df.y, df.z)
    ax.set_xlabel('Distance from start of flume (mm)')
    ax.set_ylabel('Lateral position (mm)')
    ax.set_zlabel('Vertical position (mm)')
    
    if ('limits' in keyword_parameters):
        limits=keyword_parameters['limits']
        ax.set_xlim(limits[0],limits[1])
        ax.set_ylim(limits[2],limits[3])
        ax.set_zlim(limits[4],limits[5])
    plt.show()
    
def smooth(df,columns,names,period,**keyword_parameters):
    """Smooths out spikey data in a 3D trajectory by running a moving average
    over specified columns of input dataframe. Optionally, the smoothed curve
    can be shifted back close to its originally position by including the shift
    optional argument.
    
    Args:
    df      (dataframe): pandas dataframe
    columns      (list): list of dataframe headers to smooth
    names        (list): names of new columns in same order as columns
    
    Returns:
    dataframe: pandas dataframe containing smoothed and/or shifted columns.
    """
    df=deepcopy(df)
    for i in range(len(columns)):
        df[names[i]]=pd.rolling_mean(df[columns[i]],period)
            
    if ('shift' in keyword_parameters):
        shift = keyword_parameters['shift']
    
    if shift:
        shift_names=keyword_parameters['shift_names']
        shift_period=keyword_parameters['shift_period']
        
        for i in range(len(columns)):
            df[shift_names[i]]=df[names[i]].shift(shift_period)   
    return df

def calculate_vel_components(df,columns,names,fps,**keyword_parameters):
    """Add velocity components of 3D track to input dataframe. Optionally adds
    the velocity magnitude ('V').
    
    Args:
    df      (dataframe): pandas dataframe
    columns      (list): list of dataframe headers to smooth
    names        (list): names of new columns in same order as columns
    
    Returns:
    dataframe: pandas dataframe containing velocity component and/or magnitude.
    """

    df=deepcopy(df)
       
    for i in range(len(columns)):
         df[names[i]]=(df[columns[i]].diff())*fps
    
    if ('mag' in keyword_parameters):
         mag = keyword_parameters['mag']
         if mag:
              df['V']=(df[names[0]]**2+df[names[1]]**2+df[names[2]]**2)**0.5
    if ('acceleration' in keyword_parameters):
         acc = keyword_parameters['acceleration']
         if acc:
             for i in range(len(columns)):
                 name=names[i]+'_acc'
                 df[name]=df[names[i]].diff()
    
    return df


def check_in_3D_zone(df):
    
    df=deepcopy(df)
    df['Long_zone']=np.nan
    df['Lat_zone']=np.nan
    df['Vert_zone']=np.nan
    #c2
    df['Long_zone']=np.where((df.x >= 1340),'fore',df['Long_zone']) 
    df['Long_zone']=np.where(((df.x >= 1040) & (df.x <= 1200)),'fore', df['Long_zone']) 
    df['Long_zone']=np.where(((df.x > 1200) & (df.x < 1340)),'aft', df['Long_zone']) 
    df['Long_zone']=np.where((df.x < 1040),'aft', df['Long_zone']) 
    
    df['Lat_zone']=np.where((df.y < 152),'right_wall',df['Lat_zone']) 
    df['Lat_zone']=np.where((df.y <= 120),'right',df['Lat_zone']) 
    df['Lat_zone']=np.where((df.y <= 75 ),'left',df['Lat_zone']) 
    df['Lat_zone']=np.where((df.y <= 30 ),'left_wall',df['Lat_zone']) 
    
    df['Vert_zone']=np.where((df.z < 90),'high',df['Vert_zone']) 
    df['Vert_zone']=np.where((df.z <= 60),'mid',df['Vert_zone']) 
    df['Vert_zone']=np.where((df.z <= 25 ),'low',df['Vert_zone']) 

    df['Zone']=np.nan
    df['Zone']=np.where(((df.Long_zone == 'fore')& (df.Lat_zone == 'left_wall')& (df.Vert_zone == 'low')),1,df['Zone'])
    df['Zone']=np.where(((df.Long_zone == 'fore')& (df.Lat_zone == 'left')& (df.Vert_zone == 'low')),2,df['Zone']) 
    df['Zone']=np.where(((df.Long_zone == 'fore')& (df.Lat_zone == 'right')& (df.Vert_zone == 'low')),3,df['Zone']) 
    df['Zone']=np.where(((df.Long_zone == 'fore')& (df.Lat_zone == 'right_wall')& (df.Vert_zone == 'low')),4,df['Zone']) 
    df['Zone']=np.where(((df.Long_zone == 'aft')& (df.Lat_zone == 'left_wall')& (df.Vert_zone == 'low')),5,df['Zone'])
    df['Zone']=np.where(((df.Long_zone == 'aft')& (df.Lat_zone == 'left')& (df.Vert_zone == 'low')),6,df['Zone']) 
    df['Zone']=np.where(((df.Long_zone == 'aft')& (df.Lat_zone == 'right')& (df.Vert_zone == 'low')),7,df['Zone']) 
    df['Zone']=np.where(((df.Long_zone == 'aft')& (df.Lat_zone == 'right_wall')& (df.Vert_zone == 'low')),8,df['Zone']) 
    
    df['Zone']=np.where(((df.Long_zone == 'fore')& (df.Lat_zone == 'left_wall')& (df.Vert_zone == 'mid')),9,df['Zone'])
    df['Zone']=np.where(((df.Long_zone == 'fore')& (df.Lat_zone == 'left')& (df.Vert_zone == 'mid')),10,df['Zone']) 
    df['Zone']=np.where(((df.Long_zone == 'fore')& (df.Lat_zone == 'right')& (df.Vert_zone == 'mid')),11,df['Zone']) 
    df['Zone']=np.where(((df.Long_zone == 'fore')& (df.Lat_zone == 'right_wall')& (df.Vert_zone == 'mid')),12,df['Zone']) 
    df['Zone']=np.where(((df.Long_zone == 'aft')& (df.Lat_zone == 'left_wall')& (df.Vert_zone == 'mid')),13,df['Zone'])
    df['Zone']=np.where(((df.Long_zone == 'aft')& (df.Lat_zone == 'left')& (df.Vert_zone == 'mid')),14,df['Zone']) 
    df['Zone']=np.where(((df.Long_zone == 'aft')& (df.Lat_zone == 'right')& (df.Vert_zone == 'mid')),15,df['Zone']) 
    df['Zone']=np.where(((df.Long_zone == 'aft')& (df.Lat_zone == 'right_wall')& (df.Vert_zone == 'mid')),16,df['Zone']) 
    
    df['Zone']=np.where(((df.Long_zone == 'fore')& (df.Lat_zone == 'left_wall')& (df.Vert_zone == 'high')),17,df['Zone'])
    df['Zone']=np.where(((df.Long_zone == 'fore')& (df.Lat_zone == 'left')& (df.Vert_zone == 'high')),18,df['Zone']) 
    df['Zone']=np.where(((df.Long_zone == 'fore')& (df.Lat_zone == 'right')& (df.Vert_zone == 'high')),19,df['Zone']) 
    df['Zone']=np.where(((df.Long_zone == 'fore')& (df.Lat_zone == 'right_wall')& (df.Vert_zone == 'high')),20,df['Zone']) 
    df['Zone']=np.where(((df.Long_zone == 'aft')& (df.Lat_zone == 'left_wall')& (df.Vert_zone == 'high')),21,df['Zone'])
    df['Zone']=np.where(((df.Long_zone == 'aft')& (df.Lat_zone == 'left')& (df.Vert_zone == 'high')),22,df['Zone']) 
    df['Zone']=np.where(((df.Long_zone == 'aft')& (df.Lat_zone == 'right')& (df.Vert_zone == 'high')),23,df['Zone']) 
    df['Zone']=np.where(((df.Long_zone == 'aft')& (df.Lat_zone == 'right_wall')& (df.Vert_zone == 'high')),24,df['Zone']) 


    
    
#    long=['fore','aft']
#    lat=['right_wall','right','left','left_wall']
#    vert=['low','mid','high']
    
            
    return df
    

def check_time_in_zones(df):

#    df=deepcopy(df)
#    df=df.groupby(['Zone']).count()
    df=df.join(df.groupby('Zone')['Zone'].count(), on='Zone', rsuffix='_counts')
#    df=df.reset_index()
    
    return df


def check_direction(df,limits,names):
    
    df=deepcopy(df)
    df['direction']=np.nan
    df['direction']=np.where(((df.x.iloc[0] <= limits[0]) & (df.x.iloc[-1] >= limits[1] )),names[0], df.direction)
    df['direction']=np.where(((df.x.iloc[0] >= limits[1]) & (df.x.iloc[-1] <= limits[0] )),names[1], df.direction)
    df['direction']=np.where(((df.x.iloc[0] <= limits[0]) & (df.x.iloc[-1] <= limits[0] )),names[2], df.direction)
    df['direction']=np.where(((df.x.iloc[0] >= limits[1]) & (df.x.iloc[-1] >= limits[1] )),names[3], df.direction)
    
    return df
    