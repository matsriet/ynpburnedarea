# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 20:26:40 2021

@author: matsr
"""

# Import packages
import pandas as pd
import numpy as np
import os
from scipy.interpolate import interp1d

def read_eco(path, label):
    """ 
    Reads spectral files from NASA's ecostress spectral library (used for substrate endmembers).

    Parameters
    ----------
    path : str
        Path of the sample txt file.
    label : str
        Label to assign to the sample.

    Returns
    -------
    df : pandas dataframe
        Dataframe containing reflectance values indexed by wavelength in nm.

    """
    
    df = pd.read_csv(path, names=['wavelength', label], skiprows=20, sep='\t', usecols=[0,1]) #Read the csv file
    df[label] = df[label]/100 #Divide reflectances by 100 (percentages to fractions)
    wl_nm = np.int_(np.round(np.array(df['wavelength']*1000))) #Convert wavelengths in um to nm and remove decimals to obtain wavelengths
    df = df.set_index(wl_nm) #Set index to the wavelength in nm
    df = df.drop('wavelength', axis=1) #Drop the unused wavelength in um
    return df

def read_avi(path, label, wl_path=r'C:\Users\matsr\OneDrive\Desktop\MSc_Project\spectral_data\YNP_v2\splib07a_Wavelengths_AVIRIS_1996_0.37-2.5_microns.txt'):
    """
    Reads spectral files from aviris files from the USGS spectral libary (used for vegetation endmembers).

    Parameters
    ----------
    path : str
        Path of the sample txt file.
    label : str
        Label to assign to the sample.
    wl_path : str
        Path of the wavelength file of the Aviris samples. 
        Name of the file is: splib07a_Wavelengths_AVIRIS_1996_0.37-2.5_microns.txt

    Returns
    -------
    df : pandas dataframe
        Dataframe containing reflectance values indexed by wavelength in nm.

    """
    
    df = pd.read_csv(path, names=[label], skiprows=1)
    wl = pd.read_csv(wl_path, names=['wavelength'], skiprows=1)
    wl_nm = np.int_(np.round(np.array(wl['wavelength']*1000))) #Convert wavelengths in um to nm and remove decimals to obtain wavelengths
    df = df.set_index(wl_nm) #Set index to the wavelength in nm
    df = df[df[label]>=0] #Remove NaN values
    return df
      
def spectral_response(spectrum, rsr_nir, rsr_swir):
    """
    Calculates NIR and SWIR values given a dataframe of reflectance and satellite response dataframes.

    Parameters
    ----------
    spectrum : pandas dataframe
        Dataframe containing reflectance of a spectral endmember sample. 
        Can be obtained from read_eco or read_avi functions.
    rsr_nir : pandas dataframe
        Dataframe containing spectral response function in the NIR spectral domain of a satellite.
    rsr_swir : pandas dataframe
        DESCRIPTION.

    Returns
    -------
    nir : TYPE
        NIR response of the sample for the satellite.
    swir : TYPE
        SWIR response of the sample for the satellite.
    """
    
    #Obtain wavelength ranges of the response function
    nir_wl = np.array(rsr_nir.index)
    swir_wl = np.array(rsr_swir.index)
    
    spec_wl = np.array(spectrum.index)
    spec = np.array(spectrum).flatten()
    
    #Interpolate spectrum at instrument wavelength ranges
    spec_nir = interpolate(spec_wl, spec, nir_wl)
    spec_swir = interpolate(spec_wl, spec, swir_wl)
    
    #Calculate band responses
    nir = sum(spec_nir * rsr_nir)/rsr_nir.sum()
    swir = sum(spec_swir * rsr_swir)/rsr_swir.sum()
    
    
    return nir, swir
    
def interpolate(x, y, xtarget):
    #Interpolates y on xtarget
    f = interp1d(x, y)
    ytarget = f(xtarget)
    
    return ytarget
    

def listdir(directory, source='eco'):
    #Gives a dataframe listing the files in the directory.
    
    #Initialise list
    paths = []  
    names = []
    
    #Loop over files in directory and add them to list
    for entry in os.scandir(directory):
        if entry.is_file():
            paths.append(entry.path)
            names.append(entry.name)
    
    data = pd.DataFrame(names, columns=['File']) #Convert to dataframe
    data['Path'] = paths
    return data

