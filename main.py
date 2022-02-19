# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 13:53:32 2021

@author: matsr
"""

from scipy.io import loadmat
import numpy as np
import pandas as pd
import func
import os
import pickle
import xarray as xr
import matplotlib.pyplot as plt

#%% Define functions
def em_data(folder_path, rsr_nir, rsr_swir, source):
    '''
    Converts sets of spectral datasets to minimal, averaged and maximal satellite responses.

    Parameters
    ----------
    folder_path : string
        Path of a folder containing other folders with sets of samples. 
        The sample sets will be averaged to result in satellite responses.
    rsr_nir : pandas DataFrame
        Dataframe containing satellite response functions for the NIR spectral domain.
    rsr_swir : pandas DataFrame
        Dataframe containing satellite response functions for the SWIR spectral domain.
    source : string
        sample data source. Valid sources are:
            'eco' for JPL's ecospeclib data.
            'avi' for aviris data.
        
    Returns
    -------
    Three pandas dataframes containing minimum, averages and maximums for the sample sets.

    '''
    
    #Obtain column names for the dataframes
    column_names = ['Folder']
    for column in rsr_nir:
        column_names.append(column + '_nir')
        column_names.append(column + '_swir')
    
    #Initialize dataframe for storing the data
    em_data = pd.DataFrame(columns = column_names)
    
    #Loop over the items in the folders
    for folder in os.scandir(folder_path):
        #print(folder.name)
        
        for index, entry in enumerate(os.scandir(folder)):
            if entry.is_file():
                print('file read:', entry.name)
                #Obtain spectral reflectance for this sample
                if source == 'eco':
                    if entry.name.endswith('ancillary.txt'):
                        continue
                    df = func.read_eco(entry.path, label=entry.name)
                elif source == 'avi':
                    df = func.read_avi(entry.path, label=entry.name)
                    
                    
                response_list = [folder.name] #Create a list to save responses for this entry
                
                #Calculate spectral responses
                for column in rsr_nir:
                    response = func.spectral_response(df, rsr_nir[column], rsr_swir[column])
                    response_list = response_list + list(response)
                response_series = pd.Series(response_list, index=column_names, name=(entry.name))
                
                em_data = em_data.append(response_series)
            
    return em_data

def find_fburn(valid_range=[0,1]):
    #Calculates burned fraction directly. See appendix A for more info.
    
    #Calculate individual terms of the equations
    N = nbr_start - dNBR_thresh
    kg = frac*(1-delta_char)
    
    Rvpos = veg_nir + veg_swir #Rv+
    Rvneg = veg_nir - veg_swir #Rv-
    Rgpos = gr_nir + gr_swir #Rg+
    Rgneg = gr_nir - gr_swir #Rg-
    Rcpos = char_nir + char_swir #Rc+
    Rcneg = char_nir - char_swir #Rc-
     
    delim = frac*(Rvneg - N*Rvpos) + (Rgneg - N*Rgpos)*(1-frac) #Delimiter of the equation
    denom = frac*(Rvneg - N*Rvpos) + kg*(N*Rgpos - Rgneg) + frac*delta_char*(N*Rcpos-Rcneg) #Denominator
    
    # Calculate f_burn_intercept
    fb_intercept = delim/denom
    
    if fb_intercept > valid_range[1]:
        fburn = 1
    elif fb_intercept < valid_range[0]:
        fburn = 1
    else:
        fburn = fb_intercept
    
    return fburn
        

def check_fburn():
    #Calculate fburn using a (very slow) iterative method. Used to check direct calculation for bugs
    
    burn_increment = 0.01
    burn_range = np.arange(burn_increment, 1 + burn_increment, burn_increment) #For speed, maybe later change this to while loop
    
    for b, burn in enumerate(burn_range): #increase burned fraction                                   
        #Calculate spectral contributions
        frac_veg = (1-burn)*frac   #Lower the contribution of vegetation linearly
        frac_char = burn*frac*delta_char 
        frac_ground = 1 - frac_veg - frac_char

        #Check if there is enough ground contribution to convert to charcoal (only required for delta_char>1)
        if frac_ground <= 0:
            frac_char = 1 - frac_veg #If not, merely convert vegetation to charcoal
            frac_ground = 0 #And set frac_ground to stay at 0
        
        #Calculate NBR
        nir = gr_nir*frac_ground + veg_nir*frac_veg + char_nir*frac_char
        swir = gr_swir*frac_ground + veg_swir*frac_veg + char_swir*frac_char
        nbr = (nir - swir)/(nir + swir)
        dnbr = nbr_start - nbr
    
        if dnbr >= dNBR_thresh:
            #print('detected at burn%', burn)
            break
        
    return(burn)

def output_fburn(fburn, only_nbr=False):
    #Calculate spectral contributions
    frac_veg = (1-fburn)*frac
    frac_char = fburn*frac*delta_char 
    frac_ground = 1 - frac_veg - frac_char
   
    #Calculate NBR
    nir = gr_nir*frac_ground + veg_nir*frac_veg + char_nir*frac_char
    swir = gr_swir*frac_ground + veg_swir*frac_veg + char_swir*frac_char
    nbr = (nir - swir)/(nir + swir)

    if only_nbr:
        return(nbr)
    else:
        return(frac_veg, frac_char, frac_ground, nir, swir, nbr)


#%% Obtain Spectral responses

#Response functions of satellite instruments
rsr_nir = pd.read_csv(r'spectral_responses_overview_SWIR.csv', index_col = 0)
rsr_swir = pd.read_csv(r'spectral_responses_overview_SWIR.csv', index_col = 0)

#Responses of charcoal
char1 = loadmat(r'CHAR1.mat')['sp'] #You may want to add folder path
char2 = loadmat(r'CHAR2.mat')['sp']
char3 = loadmat(r'CHAR3.mat')['sp']

char_wl = np.int_(np.round(char1[:,0]*1000)) #Convert wavelengths in um to nm and remove decimals to obtain wavelengths

#Combine charcoal reflectances
char = pd.DataFrame(char1[:,1], index=char_wl)
char[1] = char2[:,1]
char[2] = char3[:,1]

#Obtain spectral response for charcoal
charcoal = pd.DataFrame(char.columns, columns = ['Species'])

for column in rsr_nir:
    charcoal[column + '_nir'] = ''
    charcoal[column + '_swir'] = ''
    for name, values in char.iteritems():
        response = func.spectral_response(values, rsr_nir[column], rsr_swir[column])
        charcoal[column + '_nir'][name] = response[0]
        charcoal[column + '_swir'][name] = response[1]


# Obtain responses of vegetation
veg_data = em_data(r'', rsr_nir, rsr_swir, source='avi') #Input here the folder path of the vegetation samples from aviris

# Obtain responses of geological unit
gr_data = em_data(r'', rsr_nir, rsr_swir, source='eco') #Input here the folder path of the substrate samples from ecospeclib

#%% Analysis settings

#Setting the resolutions of the model
dNBR_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25] #Set the dNBR thresholds to test. 
veg_increment = 0.05 #Step size of increasing the vegetation area fraction for the samples (0 < x < 1)
veg_range = np.arange(veg_increment, 1 + veg_increment, veg_increment)


#Burn landscape evolution settings
delta_chars = [0, 0.25, 0.5, 0.75, 1]

'''
The delta_char affect how a decrease in forest cover is reflected in the changes in endmember contributions.
Gives the % of charcoal cover change per % vegetation loss. Should be between 0 and 2.
x = 0:  models devegetation without charcoal production; vegetation cover is converted to bare soil.
0<x<1:  models mixture of charcoal accumulation and soil exposure.
x = 1:  models pure charcoal accumulation; all of the vegetation spectral contribution is converted to charcoal.
1<x<2:  vegetation and soil contributions are converted to charcoal cover.
x = 2:  for every % vegetation contribution loss, a % of the pixel are in soil is also converted to charcoal.
'''

#%% Main

#Find folder names
gr_folders = gr_data['Folder'].value_counts().index.sort_values()
veg_folders = veg_data['Folder'].value_counts().index.sort_values()

#dimensions to save variables for. [dchar, dnbr, instruments, ground folders, veg folders, min/mean/max, f_veg_preburn]
dims = [len(delta_chars), len(dNBR_thresholds), len(rsr_nir.columns), len(gr_folders), len(veg_folders), 3, len(veg_range)]

#Initialise arrays
burn_frac = np.full(dims, np.nan)
char_contr = np.full(dims, np.nan)
veg_contr = np.full(dims, np.nan)
ground_contr = np.full(dims, np.nan)
NBR_preburn = np.full(dims, np.nan)
NBR_ondetec = np.full(dims, np.nan)
undetected = np.full(dims, np.nan)

n_samples = 0
burn_test = []
burn_check_test = []

print('Starting burn analysis')
for dc, delta_char in enumerate(delta_chars):
    print('Delta_char was set to', delta_char)
    
    for d, dNBR_thresh in enumerate(dNBR_thresholds):
        print('Calculating for dNBR threshold', dNBR_thresh, ':')
        
        for g, gr_folder in enumerate(gr_folders): #Loop over every folder in ground data
            print(gr_folder)
            gr_slice = gr_data[gr_data['Folder'] == gr_folder]
            
            for v, veg_folder in enumerate(veg_folders): #Loop over vegetation folder
                veg_slice = veg_data[veg_data['Folder'] == veg_folder]  
                print('---' + veg_folder)
                
                for i, instrument in enumerate(rsr_nir.columns): #Loop over instruments
                    for f, frac in enumerate(veg_range):  
                        
                        #Initialise lists that temporarily save the data for every sample
                        burn_frac_list = []
                        char_contr_list = []
                        veg_contr_list = []
                        ground_contr_list = []
                        NBR_ondetec_list = []
                        NBR_preburn_list = []
                        undetected_list = []
                             
                        #Now loop over every sample
                        for gs, gr_sample in gr_slice.iterrows(): #Loop over the ground sample
                            gr_nir = gr_sample[instrument + '_nir']
                            gr_swir = gr_sample[instrument + '_swir']
                            for vs, veg_sample in veg_slice.iterrows(): #Loop over the vegetation samples
                                veg_nir = veg_sample[instrument + '_nir']
                                veg_swir = veg_sample[instrument + '_swir']
                                
                                #Calculate starting NBR (as this is invariant to char)
                                nir_s = gr_nir*(1-frac) + veg_nir*frac
                                swir_s = gr_swir*(1-frac) + veg_swir*frac
                                nbr_start = (nir_s - swir_s) / (nir_s + swir_s)
                                
                                for c, char in charcoal.iterrows(): #Loop over the charcoal samples
                                    char_nir = char[instrument + '_nir']
                                    char_swir = char[instrument + '_swir']
                                    
                                    n_samples += 1
                                    
                                    burn = find_fburn()
                                    
                                    
                                    if burn == 1:
                                        undetected_list.append(1)
                                    else:
                                        undetected_list.append(0)
                                
                                    # Sample a few of the samples and test them with iteration. Comment out not to check, saves computation time
                                    '''
                                    if n_samples % 1000 == 0:
                                        burn_check = check_fburn()
                                        burn_test.append(burn)
                                        burn_check_test.append(burn_check)
                                        if abs(burn - burn_check) > 0.01:
                                            print('burn calculations differ too much')
                                            print(n_samples, burn, burn_check)
                                    '''
                                    #Calculate spectral contributions
                                    frac_veg, frac_char, frac_ground, nir, swir, nbr = output_fburn(burn)
                                    
                                    #Here you would test if frac_ground > 0
                                    #Model otherwise the groundless detectability. 
                                    #But instead, print an error explaining that it is not implemented.
                                    if delta_char > 1:
                                        print('ERROR, delta_char > 1 is not supported')
                                        exit()
        
                                    #Save data to lists
                                    burn_frac_list.append(burn)
                                    char_contr_list.append(frac_char)
                                    veg_contr_list.append(frac_veg)
                                    ground_contr_list.append(frac_ground)
                                    NBR_ondetec_list.append(nbr)
                                    NBR_preburn_list.append(nbr_start)
                        
                        #Save data to arrays
                        burn_frac[dc,d,i,g,v,:,f] = [np.array(burn_frac_list).min(), np.array(burn_frac_list).mean(), np.array(burn_frac_list).max()]
                        char_contr[dc,d,i,g,v,:,f] = [np.array(char_contr_list).min(), np.array(char_contr_list).mean(), np.array(char_contr_list).max()]
                        veg_contr[dc,d,i,g,v,:,f] = [np.array(veg_contr_list).min(), np.array(veg_contr_list).mean(), np.array(veg_contr_list).max()]
                        ground_contr[dc,d,i,g,v,:,f] = [np.array(ground_contr_list).min(), np.array(ground_contr_list).mean(), np.array(ground_contr_list).max()]
                        NBR_ondetec[dc,d,i,g,v,:,f] = [np.array(NBR_ondetec_list).min(), np.array(NBR_ondetec_list).mean(), np.array(NBR_ondetec_list).max()]
                        NBR_preburn[dc,d,i,g,v,:,f] = [np.array(NBR_preburn_list).min(), np.array(NBR_ondetec_list).mean(), np.array(NBR_ondetec_list).max()]
                        undetected[dc,d,i,g,v,:,f] = [np.array(undetected_list).min(), np.array(undetected_list).mean(), np.array(undetected_list).max()]
#%% Export data xarray and convert to pickle

dims = ['delta_char', 'dNBR_thresh', 'instrument', 'ground', 'vegetation', 'measure', 'veg%']
coords = {'delta_char':delta_chars, 'dNBR_thresh':dNBR_thresholds, 'instrument':list(rsr_nir.columns), 'ground':list(gr_folders), 'vegetation':list(veg_folders), 'measure':['min', 'avg', 'max'], 'veg%':veg_range}

burn_frac = xr.DataArray(burn_frac, dims=dims, coords=coords, attrs={'long_name': 'Fraction of vegetation burned on detection'})
char_contr = xr.DataArray(char_contr, dims=dims, coords=coords, attrs={'long_name': 'Spectral contribution fraction of charcoal on detection'})
veg_contr = xr.DataArray(veg_contr, dims=dims, coords=coords, attrs={'long_name': 'Spectral contribution fraction of vegetation on detection'})
ground_contr = xr.DataArray(ground_contr, dims=dims, coords=coords, attrs={'long_name': 'Spectral contribution fraction of ground'})
NBR_ondetec = xr.DataArray(NBR_ondetec, dims=dims, coords=coords, attrs={'long_name': 'Normalised burn ratio on detection'})
NBR_preburn = xr.DataArray(NBR_preburn, dims=dims, coords=coords, attrs={'long_name': 'Normalised burn ratio before burn'})
undetected = xr.DataArray(undetected, dims=dims, coords=coords, attrs={'long_name': 'Percentage of pixels undetected'})
data = dict(burn_frac=burn_frac, char_contr=char_contr, veg_contr=veg_contr, ground_contr=ground_contr, NBR_ondetec=NBR_ondetec, NBR_preburn=NBR_preburn, undetected=undetected)
dataset = xr.Dataset(data)

#%% Save data to file
file = open(r'C:\Users\matsr\OneDrive\Desktop\MSc_Project\results_data\run3\0_precambrian_gneiss_schist.pkl', 'wb')
pickle.dump(dataset, file)# dump information to that file
file.close() # close the file

#%% Compare calculations with iteration via plot

plt.scatter(burn_test, burn_check_test, s=0.03)
plt.xlabel('fburn from calculation')
plt.ylabel('fburn from iteration')


n = 0
for i in range(len(burn_test)):
    if abs(burn_test[i] - burn_check_test[i]) > 0.01:
        print(i, burn_test[i], burn_check_test[i])
        n += 1

print(n, 'of the', i, 'tested samples show deviation between calculation and iteration')
