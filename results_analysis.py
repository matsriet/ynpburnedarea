# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 13:58:49 2021

@author: matsr
"""
import pickle
import numpy as np
import pandas as pd
import xarray as xr
import os
import matplotlib.pyplot as plt
from scipy import stats

#%% Set folder path of pickled data obtained from main.py
folder_path = r''

#%% Geology abundance settings
#geol_abun = [0.03, 0.03, 0.03, 0, 0, 0.2, 0.2, 0.3, 0.01, 0.2]
veg_contr_start = np.arange(0.05, 1.05, 0.05)
veg_contr_start = np.around(veg_contr_start, decimals=2) #Fixes weird float errors
ground_contr_start = 1-veg_contr_start # 1 minus vegetation fraction (see main script)

#%% Vegetation abundance settings
comb_abun = [[5716, 5145,   4384,   0,  58,     1101],
            [1199,  5537,   4315,   0,  325,    2917],
            [902,   7790,   5881,   0,  1984,   6346],
            [0,     13,     5,      0,  0,      11],
            [687,   669,    2002,   0,  39,     1252],
            [9962,  48663,  49002,  0,  7131,   64636],
            [6999,  91756,  4396,   0,  10702,  9266],
            [618,   159440, 14834,  0,  14709,  18173],
            [1949,  11826,  1233,   0,  384,    331],
            [9431,  113677, 69943,  0,  53359,  16703]]

comb_abun = np.array(comb_abun, dtype=float)

'''
Abundancies (%) of vegetations and the various geologies.
Order of rows is [douglas_fir, lodgepole_pine, nonforest, other, spruce_fir, whitebark_pine]
Order of colums is [0_precambrian_gneiss_schist, 1_paleozoic_formations, 2_mesozoic_formations, 3_tertairy_formations, 4_diorite_intrusions, 5_absaroka_volcanic_breccias, 6_yellowstone_tuffs, 7_plateau_rhyolite, 8_basalt_flows, 9_quarternary_deposits]
'''

#%% Agregate data for the various geology units (averages over rocks/soils)
geol_shape = (5,5,4,3,6,10,20)

geol_char_contr = np.full(geol_shape, np.nan)
geol_delta_ground = np.full(geol_shape, np.nan)
geol_burn_frac = np.full(geol_shape, np.nan)
geol_undetected = np.full(geol_shape, np.nan)
geol_nbr_preburn = np.full(geol_shape, np.nan)

geol_unit_names = []

#Loop over pickled geology unit files
for i, entry in enumerate(os.scandir(folder_path)):
    if entry.name.endswith('.pkl'):
        print(entry.name)
        geol_unit_names.append(entry.name)
        file = open(entry, 'rb')
        data = pickle.load(file)
        file.close()
        
        data = data.rename({'veg%':'f_veg_preburn'}) #veg% is problematic, as % is also used by python
        
        #Obtain data
        burn_frac = data['burn_frac']
        char_contr = data['char_contr']
        ground_contr = data['ground_contr']
        undetected = data['undetected']
        nbr_preburn = data['NBR_preburn']
        
        '''
        #Alternative for runs that do not have an undetected incorporated
        #Identify undetected pixels (where the burned fraction reached 100%)
        undetected = burn_frac.where(burn_frac==1,0)
        '''
        #Calculate change in ground contribution
        delta_ground = data['ground_contr'] #Create delta_ground with dummy data
        
        for n, gr_contr_start in enumerate(ground_contr_start):
            delta_ground[:,:,:,:,:,:,n] = ground_contr[:,:,:,:,:,:,n] - gr_contr_start
        
        #Aggradate over ground axis, taking into account uncertainty (needed as the abundances of the rocks/soils in the units is unknown)
        char_contr_min = char_contr.min(dim=('measure', 'ground'))
        char_contr_mean = char_contr.loc[dict(measure='avg')].mean(dim=('ground'))
        char_contr_max = char_contr.max(dim=('measure', 'ground'))
        
        delta_ground_min = delta_ground.min(dim=('measure', 'ground'))
        delta_ground_mean = delta_ground.loc[dict(measure='avg')].mean(dim=('ground'))
        delta_ground_max = delta_ground.max(dim=('measure', 'ground'))
        
        burn_frac_min = burn_frac.min(dim=('measure', 'ground'))
        burn_frac_mean = burn_frac.loc[dict(measure='avg')].mean(dim=('ground'))
        burn_frac_max = burn_frac.max(dim=('measure', 'ground'))
        
        undetected_min = undetected.min(dim=('measure', 'ground'))
        undetected_mean = undetected.loc[dict(measure='avg')].mean(dim=('ground'))
        undetected_max = undetected.max(dim=('measure', 'ground'))
        
        nbr_preburn_min = nbr_preburn.min(dim=('measure', 'ground'))
        nbr_preburn_mean = nbr_preburn.loc[dict(measure='avg')].mean(dim=('ground'))
        nbr_preburn_max = nbr_preburn.max(dim=('measure', 'ground'))

    #Save data for geologies
    geol_char_contr[:,:,:,0,:,i,:] = char_contr_min
    geol_char_contr[:,:,:,1,:,i,:] = char_contr_mean
    geol_char_contr[:,:,:,2,:,i,:] = char_contr_max
    
    geol_delta_ground[:,:,:,0,:,i,:] = delta_ground_min
    geol_delta_ground[:,:,:,1,:,i,:] = delta_ground_mean
    geol_delta_ground[:,:,:,2,:,i,:] = delta_ground_max
    
    geol_burn_frac[:,:,:,0,:,i,:] = burn_frac_min
    geol_burn_frac[:,:,:,1,:,i,:] = burn_frac_mean
    geol_burn_frac[:,:,:,2,:,i,:] = burn_frac_max
    
    geol_undetected[:,:,:,0,:,i,:] = undetected_min
    geol_undetected[:,:,:,1,:,i,:] = undetected_mean
    geol_undetected[:,:,:,2,:,i,:] = undetected_max
    
    geol_nbr_preburn[:,:,:,0,:,i,:] = nbr_preburn_min
    geol_nbr_preburn[:,:,:,1,:,i,:] = nbr_preburn_mean
    geol_nbr_preburn[:,:,:,2,:,i,:] = nbr_preburn_max
    

#%% Remove unused 'other' vegetation (comment out when used)
geol_char_contr = np.delete(geol_char_contr, 3, axis=4)
geol_delta_ground = np.delete(geol_delta_ground, 3, axis=4)
geol_burn_frac = np.delete(geol_burn_frac, 3, axis=4)
geol_undetected = np.delete(geol_undetected, 3, axis=4)
geol_nbr_preburn = np.delete(geol_nbr_preburn, 3, axis=4)
comb_abun = np.delete(comb_abun, 3, axis=1)

veg_names = char_contr.coords['vegetation'].values.tolist()
veg_names = veg_names[0:3] + veg_names[4:]

#%% Remake the np arrays into xarrays (for easier manipulation and plotting)

#Obtain dimensions and coordinates
coords = char_contr.drop(('ground', 'vegetation')).coords


dims = ['delta_char', 'dNBR_thresh', 'instrument', 'measure', 'vegetation', 'geol_unit', 'f_veg_preburn']
geol_unit_coords = {'geol_unit':geol_unit_names, 'vegetation':veg_names}

#Convert to xarray from np array
geol_char_contr = xr.DataArray(geol_char_contr, dims=dims, coords=coords).assign_coords(geol_unit_coords)
geol_delta_ground = xr.DataArray(geol_delta_ground, dims=dims, coords=coords).assign_coords(geol_unit_coords)
geol_burn_frac = xr.DataArray(geol_burn_frac, dims=dims, coords=coords).assign_coords(geol_unit_coords)
geol_undetected = xr.DataArray(geol_undetected, dims=dims, coords=coords).assign_coords(geol_unit_coords)
geol_nbr_preburn = xr.DataArray(geol_nbr_preburn, dims=dims, coords=coords).assign_coords(geol_unit_coords)

#%% Agregate over geology units and vegetation to obtain averages of entire park

#Define averaging function
def park_average(array):
    #Averages over the park with given abundances of geological units and vegetations
    #Tried to do this without looping, but gave weird outputs. Unfortunately is quite slow now
    shape_array = xr.full_like(array, np.nan)
    out = shape_array[:,:,:,:,0,0,:]
    
    for c in range(array.shape[0]): #Loop over delta_char output
        for n in range(array.shape[1]): #Loop over dNBR output
            for i in range(array.shape[2]): #Loop over instruments
                for m in range(array.shape[3]): #Loop over min/mean/max
                    for v in range(array.shape[-1]): #Loop over veg%
                        weighted_avg = np.average(array[c,n,i,m,:,:,v].values, weights=comb_abun.swapaxes(0,1)) #Weighted average of data
                        out[c,n,i,m,v] = weighted_avg #Save this average
    return out

ynp_char_contr = park_average(geol_char_contr)
ynp_delta_ground = park_average(geol_delta_ground)
ynp_undetected = park_average(geol_undetected)
ynp_burn_frac = park_average(geol_burn_frac)
ynp_nbr_preburn = park_average(geol_nbr_preburn)
    
#%% Plot differences between satellites
# This piece of code calculates for every geology unit + vegetation sample the instrument deviation from mean

#Settings
target_variable = geol_burn_frac #Pick a geol or ynp variable to plot (most useful is char_contr or burn_frac)
relative = False #Whether the comparison should be normalised
#titles = data.coords['instrument'].values #Set titles for the subplots
titles = ['Landsat 8', 'Sentinel-2 A', 'Sentinel-2 B', 'MODIS']
ylabel = 'fraction burned deviation' #Set label for y-axes
xlabel = 'fraction vegetation preburn'
if relative:
    ylabel = ylabel + ' (%)'

#Obtain coordinates from target_variable
instruments = target_variable.coords['instrument'].values
f_vegs = target_variable.coords['f_veg_preburn'].values


#Initialization
compared = xr.full_like(target_variable, np.nan) #Initialize array to save values compared to mean
mean = target_variable.mean(dim='instrument') #Take the average over the instrument values

#Calculate compared data
for i in instruments:
    if relative:
        compared.loc[dict(instrument=i)] = (target_variable.loc[dict(instrument=i)] - mean) / mean *100 #Scale it to percentage from mean if relative   
    else:
        compared.loc[dict(instrument=i)] = target_variable.loc[dict(instrument=i)] - mean #Calculate difference from mean for every satellite
        
compared_reach = np.nanmax(np.absolute(compared)) #Find maximal deviation from mean
plot_reach = np.ceil(100*compared_reach)/100 #Calculate plot reach to scale y-axes to

#Calculate data to plot
mins = np.full((len(instruments), len(f_vegs)), np.nan) #Saves minimal values for instrument - f_veg_preburn
maxs = np.full((len(instruments), len(f_vegs)), np.nan) #Saves maximal values
plot_data_instruments = [] #Saves non-extreme values for boxplot plotting


for i, ins in enumerate(instruments):
    plot_data_list = []
    for v, veg in enumerate(f_vegs):
        plot_data = compared.loc[dict(instrument=ins, measure='avg', f_veg_preburn=veg)].values.flatten() #Obtain data for boxplots
        plot_data = plot_data[~np.isnan(plot_data)] #Remove any nans in the data to plot.
        
        mins[i,v] = np.nanmin(compared.loc[dict(instrument=ins, f_veg_preburn=veg)].values)
        maxs[i,v] = np.nanmax(compared.loc[dict(instrument=ins, f_veg_preburn=veg)].values)
        
        plot_data_list.append(plot_data) 
    plot_data_instruments.append(plot_data_list)

# Plot data
fig, axs = plt.subplots(len(instruments), 1, sharey=True, sharex=True)
plt.subplots_adjust(hspace=0.1)
        
for i in range(len(instruments)): #For very instrument a subplot 
    
    #Fill grey between extreme extreme values
    axs[i].fill_between(veg_contr_start, mins[i], maxs[i], color='gray', alpha=0.2)
    
    #Averages in boxplot
    axs[i].boxplot(plot_data_instruments[i], positions = veg_contr_start, widths=veg_contr_start[0]*0.5, whis='range')
    
    
    #Set axes options
    axs[i].set_ylabel(ylabel)
    axs[i].set_xlim([min(veg_contr_start)-0.05, max(veg_contr_start)+0.05])
    axs[i].set_ylim([-plot_reach, plot_reach])
    axs[i].grid(which='major', axis='y')
    axs[i].legend(labels='', title=titles[i], loc='upper left')

axs[-1].set_xlabel(xlabel)




#%% Plot influence of dNBR
ins = 0 #Set the instrument number (0=Landsat8, 1 = SA, 2= SB, 3=modis)
dc = 4 #Set the delta_char setting
dnbr_thresh_labels = ynp_burn_frac.coords['dNBR_thresh'].values
xlabel = 'fraction vegetation preburn'
ylabel = '% undetectable pixels'
target_data = ynp_undetected
multip = 100

fig, ax = plt.subplots()

for d, dnbr in enumerate(dnbr_thresh_labels): #Loop over dnbr_thresholds
    plotdata = target_data[dc,d,ins,1].values #Obtain meanplot data
    ax.plot(f_vegs, plotdata*multip, marker='.')
    

ax.fill_between(f_vegs, target_data[dc,0,ins,0]*multip, target_data[dc,4,ins,2]*multip, color='gray', alpha=0.2)
ax.set_ylim([0,1*multip])
ax.legend(labels=list(dnbr_thresh_labels),title='dNBR threshold', loc='upper right')
ax.grid()
ax.set_ylabel(ylabel)
ax.set_xlabel(xlabel)



#%% Plot number of undetected pixels depending on delta_char
target_data = ynp_undetected
multip = 100
ins = 0

fig, ax = plt.subplots()

#Single plot for instrument ins
for c in range(len(undetected.coords['delta_char'].values)):
    ax.plot(dnbr_thresh_labels, target_data[c,:,ins,1].mean(dim='f_veg_preburn')*multip, marker='.')

ax.set_ylabel('% undetectable pixels')
ax.legend(labels=list(undetected.coords['delta_char'].values), title='\u0394char')       
ax.set_xlabel('dNBR threshold')
ax.fill_between(dnbr_thresh_labels, target_data[4,:,ins,0].mean(dim='f_veg_preburn')*multip, ynp_undetected[0,:,ins,2].mean(dim='f_veg_preburn')*multip, color='gray', alpha=0.2)
ax.grid()

#All instruments
fig, axs = plt.subplots(4,1, sharex=True, sharey=True)
for i, instrument in enumerate(instruments):
    for c in range(len(undetected.coords['delta_char'].values)):
        axs[i].plot(dnbr_thresh_labels, target_data[c,:,i,1].mean(dim='f_veg_preburn'), marker='.')
    axs[i].set_ylabel('% undetected')
    axs[i].legend(labels=list(undetected.coords['delta_char'].values), title='\u0394char')
    axs[i].set_title(instrument)
        
axs[-1].set_xlabel('dNBR threshold')
        
#%% Plot data for dnbr_thresh = 0.15

fig, ax = plt.subplots()
target_data = ynp_undetected #ynp_burn_frac
multip = 100 #100 for %, f_vegs for scaled with f_vegs

for c in range(len(undetected.coords['delta_char'].values)):
    ax.plot(f_vegs, target_data[c,2,ins,1]*multip, marker='.')
    #ax.fill_between(f_vegs, target_data[c,2,ins,0]*multip, target_data[c,2,ins,2]*multip, color='gray', alpha=0.2)
    
ax.fill_between(f_vegs, target_data[4,2,ins,0]*multip, target_data[0,2,ins,2]*multip, color='gray', alpha=0.2)
ax.set_xlabel('fraction vegetation preburn')
ax.set_ylabel('% undetectable pixels')
ax.legend(labels=list(undetected.coords['delta_char'].values), title='\u0394char', loc='lower left')
ax.grid()

fig, axs = plt.subplots(4,1, sharex=True, sharey=True)
for i, instrument in enumerate(instruments):
    for c in range(len(undetected.coords['delta_char'].values)):
        axs[i].plot(f_vegs, ynp_undetected[c,2,i,1]*100, marker='.')
    axs[i].set_ylabel('% undetected')
    axs[i].legend(labels=list(undetected.coords['delta_char'].values), title='\u0394char')
    axs[i].grid()
axs[-1].set_xlabel(xlabel)



#%% Geology-vegetation detectability
dc = 1
dnbr = 2
ins = 0
target = geol_char_contr
ylabel = 'f_c'
subplot_rows = 5
subplot_cols = 2

geol_names = ['Precambrian Gneiss & Schist', 'Paleozoic Formations', 'Mesozoic Formations', 'Tertairy Formations', 'Diorite Intrusions', 'Absaroca Volcanic Breccias', 'Yellowstone Tuffs', 'Plateau Rhyolite', 'Basalt Flows', 'Quartenary Deposits']
veg_names = ['Douglas Fir', 'Lodgepole Pine', 'Nonforest', 'Spruce Fir', 'Whitebark Pine']

target = target[dc,dnbr,ins,:,:,:,:].mean(dim='f_veg_preburn')

fig, axs = plt.subplots(subplot_rows,subplot_cols, sharey='col', sharex=True)
plt.subplots_adjust(wspace=0.01, hspace=0.1)

row_nr = 0

x_pos = np.arange(len(veg_names))

for g in range(len(geol_names)):
    col_nr = g % subplot_cols
    
    axs[row_nr,col_nr].plot(x_pos, target[1,:,g],marker='.')
    axs[row_nr,col_nr].fill_between(x_pos, target[0,:,g],target[2,:,g], color='gray', alpha=0.2)
    axs[row_nr,col_nr].set_xticks(x_pos)
    axs[row_nr,col_nr].set_xticklabels(veg_names)
    axs[row_nr,col_nr].legend(labels='', title=geol_names[g])
    axs[row_nr,0].set_ylabel(ylabel)
    
    if col_nr == subplot_cols -1:
        row_nr += 1


