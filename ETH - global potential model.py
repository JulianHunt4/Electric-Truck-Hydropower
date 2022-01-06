# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 16:19:26 2021

@author: julia
"""
#%matplotlib qt

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import numpy as np
import xarray as xr
import math

from pathlib import Path
import matplotlib.cm as cm1
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature
import matplotlib.pyplot as plt2
import matplotlib.style
import matplotlib.colors as mcolors
from datetime import datetime
import cartopy.crs as ccrs
import cartopy

import itertools as it
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import geopy.distance
from math import sin, cos, sqrt, atan2, radians

#%%
#ascii_grid = np.loadtxt("C:/Users/julia/Downloads/GRIP4_density_tp1/grip4_area_land_km2.asc", skiprows=6)

#%%

#ascii_grid = np.loadtxt("C:/Users/julia/Downloads/GRIP4_density_tp1/grip4_tp1_dens_m_km2.asc", skiprows=6)

ascii_grid = np.loadtxt("C:/Users/julia/Downloads/GRIP4_density_total (1)/grip4_total_dens_m_km2.asc", skiprows=6)

#%%


roads = np.zeros(shape=(2160,4320))

count_y = 0
while count_y<2160:
 count_x = 0
 while count_x<4320:
  if ascii_grid[count_y,count_x] <= 0:
      roads[count_y,count_x] = np.nan   
  elif math.log(ascii_grid[count_y,count_x],10) < 1:
      roads[count_y,count_x] = np.nan
  else:
      roads[count_y,count_x] = math.log(ascii_grid[count_y,count_x],10)
  count_x = count_x +1
 count_y = count_y +1
        
plt2.imshow(roads)


#%%


roads_low = np.zeros(shape=(120,360))

count_y = 0
while count_y<120:
 count_x = 0
 while count_x<360:
   if np.mean(roads[(count_y+30)*12:(count_y+30)*12+12,count_x*12:count_x*12+12]) < 0.5:
      roads_low[count_y,count_x] = np.nan
   elif np.mean(roads[(count_y+30)*12:(count_y+30)*12+12,count_x*12:count_x*12+12]) == 0:
      roads_low[count_y,count_x] = np.nan
   else:
#      roads_low[count_y,count_x] = math.log(1+*+np.mean(roads[(count_y+30)*12:(count_y+30)*12+12,count_x*12:count_x*12+12]),10)
      roads_low[count_y,count_x] = np.mean(roads[(count_y+30)*12:(count_y+30)*12+12,count_x*12:count_x*12+12])

   count_x = count_x +1
 count_y = count_y +1

plt2.imshow(roads_low)

#%%

topography = np.load('C:/Users/julia/Documents/Julian/IIASA/Electric Truck Hydropower/topography.npy')

top = np.zeros(shape=(1440,4320))

 
count_y = 0
while count_y<1440:
 count_x = 0
 while count_x<4320:
   if topography[count_y,count_x] <=0 :
    top[count_y,count_x] = np.nan  
   else:    
    top[count_y,count_x] = topography[count_y,count_x]
   count_x = count_x +1
 count_y = count_y +1
 

plt2.imshow(top)

#plt2.imshow(topography)


#%%

top_low = np.zeros(shape=(120,360))
count_y = 0
while count_y<120:
 count_x = 0
 while count_x<360:
   if np.mean(topography[(count_y)*12:(count_y)*12+12,count_x*12:count_x*12+12]) < 0.5:
      top_low[count_y,count_x] = np.nan
   elif np.mean(topography[(count_y)*12:(count_y)*12+12,count_x*12:count_x*12+12]) == 0:
      top_low[count_y,count_x] = np.nan
   else:
#      roads_low[count_y,count_x] = math.log(1+*+np.mean(roads[(count_y+30)*12:(count_y+30)*12+12,count_x*12:count_x*12+12]),10)
      top_low[count_y,count_x] = np.mean(topography[(count_y)*12:(count_y)*12+12,count_x*12:count_x*12+12])
   count_x = count_x +1
 count_y = count_y +1

plt2.imshow(top_low)

#%%

#ds = xr.open_dataarray('C:/Users/julia/Documents/Julian/IIASA/Electric Truck Hydropower/orchidee_hadgem2-es_ewembi_rcp26_nosoc_2005co2_qs_global_daily_2021_2030.nc')

ds = xr.open_dataarray('C:/Users/julia/Documents/Julian/IIASA/Electric Truck Hydropower/adaptor.mars.internal-1626475738.5222895-11866-10-1c1ef29a-9209-4817-ac4c-d378c71f8f27.nc')


rain = np.zeros(shape=(1801,3600))

count_t = 0
while count_t<12:
 dstd = ds.sel(time=ds.time[count_t].values)
 dsom = dstd.values.copy()  
 count_y = 0
 while count_y<1801:
  count_x = 0
  while count_x<3600:
   if count_x<1800:
       if  np.isnan(dsom[count_y,count_x]):
         rain[count_y,count_x+1800] = np.nan
       else:
         rain[count_y,count_x+1800] = rain[count_y,count_x+1800] + dsom[count_y,count_x]        
   else:
       if  np.isnan(dsom[count_y,count_x]):
         rain[count_y,count_x-1800] = np.nan   
       else:
         rain[count_y,count_x-1800] = rain[count_y,count_x-1800] + dsom[count_y,count_x]
   count_x = count_x +1
  count_y = count_y +1
 count_t= count_t +1
 print(count_t)
 
plt2.imshow(rain)

plt2.imshow(dsom)
#%%
rain2 = np.zeros(shape=(1801,3600))

count_y = 0
while count_y<1801:
 count_x = 0
 while count_x<3600:
  if np.isnan(rain[count_y,count_x]) or rain[count_y,count_x] <= 0 :
   rain2[count_y,count_x] = np.nan
  else:
   rain2[count_y,count_x] = math.log(rain[count_y,count_x]*1000,10)
  count_x = count_x +1
 count_y = count_y +1
 
plt2.imshow(rain2)

#%%

rain_low = np.zeros(shape=(120,360))

count_y = 0
while count_y<120:
 count_x = 0
 while count_x<360:
   if np.amax(rain[(count_y+30)*10:(count_y+30)*10+10,count_x*10:count_x*10+10]) < 0:
      rain_low[count_y,count_x] = np.nan
   elif np.amax(rain[(count_y+30)*10:(count_y+30)*10+10,count_x*10:count_x*10+10]) == 0:
      rain_low[count_y,count_x] = np.nan
   else:
#      rain_low[count_y,count_x] = math.log(1+np.amax(rain[(count_y+30)*10:(count_y+30)*10+10,count_x*10:count_x*10+10]),10)
      rain_low[count_y,count_x] = np.sum(rain[(count_y+30)*10:(count_y+30)*10+10,count_x*10:count_x*10+10])

   count_x = count_x +1
 count_y = count_y +1

plt2.imshow(rain_low)


#%%

pot = np.zeros(shape=(1440,4320))

 
count_y = 0
while count_y<1440:
 count_x = 0
 while count_x<4320:
   pot[count_y,count_x] = topography[count_y,count_x]*roads[count_y+360,count_x]*rain[int((count_y+360)*3600/4320),int(count_x*3600/4320)]
   count_x = count_x +1
 count_y = count_y +1
 
plt2.imshow(pot)


#%%

pot_low = np.zeros(shape=(120,360))

count_y = 0
while count_y<120:
 count_x = 0
 while count_x<360:
   if np.amax(1+pot[count_y*12:count_y*12+12,count_x*12:count_x*12+12]) < 0:
      pot_low[count_y,count_x] = 0
   elif np.amax(1+pot[count_y*12:count_y*12+12,count_x*12:count_x*12+12]) == 0:
      pot_low[count_y,count_x] = 0
   else:
      pot_low[count_y,count_x] = math.log(1+np.amax(pot[count_y*12:count_y*12+12,count_x*12:count_x*12+12]),10)
   count_x = count_x +1
 count_y = count_y +1

plt2.imshow(pot_low)


#%%

pot2 = np.zeros(shape=(1440,4320))
minimum_cost = np.zeros(shape=(1440,4320))
road_slope = np.zeros(shape=(1440,4320))
theorétical_road_slope = np.zeros(shape=(1440,4320))
road_potential_map = np.zeros(shape=(1440,4320))
water_road_potential_map = np.zeros(shape=(1440,4320))
water_potential_map = np.zeros(shape=(1440,4320))

percent_of_water_extracted = 0.1 
#area_m2 = 9*1000*1000
#area_correction = 1

# The road_curves index adjust the difference in topography to take into account curves in the roads. 
# The road_curves varies with the different in height in the topografy
road_curves = 1

# Potential in TWh per year

count_y = 1
while count_y<1440-1:
 count_x = 1
 location_latitude = 60-count_y*0.0833333
 location_longitude = -180+count_x*0.0833333
 coords_lat_1 = (location_latitude, location_longitude)
 coords_lat_2 = (location_latitude-0.0833333, location_longitude)
 coords_lon_1 = (location_latitude, location_longitude)
 coords_lon_2 = (location_latitude, location_longitude+0.0833333)    
 straight_vertical_road_distance = geopy.distance.distance(coords_lat_1, coords_lat_2).km
 straight_horizontal_road_distance = geopy.distance.distance(coords_lon_1, coords_lon_2).km
 diagonal_road_distance = ((straight_vertical_road_distance)**2+(straight_horizontal_road_distance)**2)**0.5 
  
 while count_x<4320-1:
  minimum_cost_1 = 200
  minimum_cost_2 = 200
  minimum_cost_3 = 200
  minimum_cost_4 = 200
  minimum_cost_5 = 200
  minimum_cost_6 = 200
  minimum_cost_7 = 200
  minimum_cost_8 = 200

  road_slope_applied_1 = 0
  road_slope_applied_2 = 0
  road_slope_applied_3 = 0
  road_slope_applied_4 = 0
  road_slope_applied_5 = 0
  road_slope_applied_6 = 0
  road_slope_applied_7 = 0
  road_slope_applied_8 = 0
  
  theorétical_road_slope_1 = 0
  theorétical_road_slope_2 = 0
  theorétical_road_slope_3 = 0
  theorétical_road_slope_4 = 0
  theorétical_road_slope_5 = 0
  theorétical_road_slope_6 = 0
  theorétical_road_slope_7 = 0
  theorétical_road_slope_8 = 0
  
  water_flow_required_1 = 0
  water_flow_required_2 = 0
  water_flow_required_3 = 0
  water_flow_required_4 = 0
  water_flow_required_5 = 0
  water_flow_required_6 = 0
  water_flow_required_7 = 0
  water_flow_required_8 = 0
  
  road_potential_per_water_1 = 0
  road_potential_per_water_2 = 0
  road_potential_per_water_3 = 0
  road_potential_per_water_4 = 0
  road_potential_per_water_5 = 0
  road_potential_per_water_6 = 0
  road_potential_per_water_7 = 0
  road_potential_per_water_8 = 0
  
  if topography[count_y,count_x] < topography[count_y-1,count_x-1]:
   altitude_difference = ((topography[count_y,count_x]-topography[count_y-1,count_x-1])**2)**0.5/1000
   horizontal_distance = diagonal_road_distance
   mínimum_theorétical_road_slope = altitude_difference/horizontal_distance*100
   road_slope_applied = 0.00000000009479256881*mínimum_theorétical_road_slope**6 - 0.00000006164989263188*mínimum_theorétical_road_slope**5 + 0.00000973488567068692*mínimum_theorétical_road_slope**4 - 0.00053230155937455900*mínimum_theorétical_road_slope**3 + 0.00174393702491216*mínimum_theorétical_road_slope**2 + 0.682497950761899*mínimum_theorétical_road_slope + 0.0329663890424854   
   
   # Road potential in GWh per year
   generation_per_truck_hour = (5.98074425279631*road_slope_applied**2 + 10.3217155154475*road_slope_applied - 5.77372144824585)/24/365/1000
   road_min = np.minimum(roads[count_y+360,count_x],roads[count_y+360-1,count_x-1])
   trucks_per_hour = 17.8571428571429*road_min**2 + 146.428571428572*road_min + 0.00000000000181898940
   road_potential_per_year = generation_per_truck_hour * trucks_per_hour * 24 * 365
   road_lengh_correction = -0.00000000000870246189*(altitude_difference*1000)**3 + 0.00000014815348746280*(altitude_difference*1000)**2 - 0.00007450553353344210*(altitude_difference*1000) + 1.43839360020761000000
   road_potential_map[count_y,count_x] = road_potential_map[count_y,count_x] + road_potential_per_year * road_lengh_correction
   water_flow_required = trucks_per_hour*33.1* 24 * 365
   water_road_potential_map [count_y,count_x] = water_road_potential_map [count_y,count_x] + water_flow_required 
   road_potential_per_water = road_potential_per_year * road_lengh_correction / water_flow_required

   # Water potential in GWh per year
   water_potential_per_year = rain[int((count_y+360)*3600/4320),int(count_x*3600/4320)]*percent_of_water_extracted*straight_vertical_road_distance*straight_horizontal_road_distance*1000*1000*1000
   water_potential_map[count_y,count_x] = water_potential_per_year

   pot2[count_y,count_x] = pot2[count_y,count_x] + road_potential_per_water * np.minimum(water_flow_required,water_potential_per_year)

   minimum_cost_1 = 0.0214764160000129*road_slope_applied**4 - 1.02397109333378*road_slope_applied**3 + 18.4185244000043*road_slope_applied**2 - 151.552444666692*road_slope_applied + 520.40174000009900000000
   road_slope_applied_1 = road_slope_applied
   theorétical_road_slope_1 = mínimum_theorétical_road_slope

  if topography[count_y,count_x] < topography[count_y-1,count_x]:
   altitude_difference = ((topography[count_y,count_x]-topography[count_y-1,count_x])**2)**0.5/1000
   horizontal_distance = diagonal_road_distance
   mínimum_theorétical_road_slope = altitude_difference/horizontal_distance*100
   road_slope_applied = 0.00000000009479256881*mínimum_theorétical_road_slope**6 - 0.00000006164989263188*mínimum_theorétical_road_slope**5 + 0.00000973488567068692*mínimum_theorétical_road_slope**4 - 0.00053230155937455900*mínimum_theorétical_road_slope**3 + 0.00174393702491216*mínimum_theorétical_road_slope**2 + 0.682497950761899*mínimum_theorétical_road_slope + 0.0329663890424854   
   
   # Road potential in GWh per year
   generation_per_truck_hour = (5.98074425279631*road_slope_applied**2 + 10.3217155154475*road_slope_applied - 5.77372144824585)/24/365/1000
   road_min = np.minimum(roads[count_y+360,count_x],roads[count_y+360-1,count_x])
   trucks_per_hour = 17.8571428571429*road_min**2 + 146.428571428572*road_min + 0.00000000000181898940
   road_potential_per_year = generation_per_truck_hour * trucks_per_hour * 24 * 365
   road_lengh_correction = -0.00000000000870246189*(altitude_difference*1000)**3 + 0.00000014815348746280*(altitude_difference*1000)**2 - 0.00007450553353344210*(altitude_difference*1000) + 1.43839360020761000000
   road_potential_map[count_y,count_x] = road_potential_map[count_y,count_x] + road_potential_per_year * road_lengh_correction
   water_flow_required = trucks_per_hour*33.1* 24 * 365
   water_road_potential_map [count_y,count_x] = water_road_potential_map [count_y,count_x] + water_flow_required    
   road_potential_per_water = road_potential_per_year * road_lengh_correction / water_flow_required

   # Water potential in GWh per year
   water_potential_per_year = rain[int((count_y+360)*3600/4320),int(count_x*3600/4320)]*percent_of_water_extracted*straight_vertical_road_distance*straight_horizontal_road_distance*1000*1000*1000
   water_potential_map[count_y,count_x] = water_potential_per_year
   
   pot2[count_y,count_x] = pot2[count_y,count_x] + road_potential_per_water * np.minimum(water_flow_required,water_potential_per_year)

   minimum_cost_2 = 0.0214764160000129*road_slope_applied**4 - 1.02397109333378*road_slope_applied**3 + 18.4185244000043*road_slope_applied**2 - 151.552444666692*road_slope_applied + 520.40174000009900000000 
   road_slope_applied_2 = road_slope_applied
   theorétical_road_slope_2 = mínimum_theorétical_road_slope


  if topography[count_y,count_x] < topography[count_y-1,count_x+1]:
   altitude_difference = ((topography[count_y,count_x]-topography[count_y-1,count_x+1])**2)**0.5/1000
   horizontal_distance = diagonal_road_distance
   mínimum_theorétical_road_slope = altitude_difference/horizontal_distance*100
   road_slope_applied = 0.00000000009479256881*mínimum_theorétical_road_slope**6 - 0.00000006164989263188*mínimum_theorétical_road_slope**5 + 0.00000973488567068692*mínimum_theorétical_road_slope**4 - 0.00053230155937455900*mínimum_theorétical_road_slope**3 + 0.00174393702491216*mínimum_theorétical_road_slope**2 + 0.682497950761899*mínimum_theorétical_road_slope + 0.0329663890424854   
   
   # Road potential in GWh per year
   generation_per_truck_hour = (5.98074425279631*road_slope_applied**2 + 10.3217155154475*road_slope_applied - 5.77372144824585)/24/365/1000
   road_min = np.minimum(roads[count_y+360,count_x],roads[count_y+360-1,count_x+1])
   trucks_per_hour = 17.8571428571429*road_min**2 + 146.428571428572*road_min + 0.00000000000181898940
   road_potential_per_year = generation_per_truck_hour * trucks_per_hour * 24 * 365
   road_lengh_correction = -0.00000000000870246189*(altitude_difference*1000)**3 + 0.00000014815348746280*(altitude_difference*1000)**2 - 0.00007450553353344210*(altitude_difference*1000) + 1.43839360020761000000
   road_potential_map[count_y,count_x] = road_potential_map[count_y,count_x] + road_potential_per_year * road_lengh_correction
   water_flow_required = trucks_per_hour*33.1* 24 * 365
   water_road_potential_map [count_y,count_x] = water_road_potential_map [count_y,count_x] + water_flow_required    
   road_potential_per_water = road_potential_per_year * road_lengh_correction / water_flow_required

   # Water potential in GWh per year
   water_potential_per_year = rain[int((count_y+360)*3600/4320),int(count_x*3600/4320)]*percent_of_water_extracted*straight_vertical_road_distance*straight_horizontal_road_distance*1000*1000*1000
   water_potential_map[count_y,count_x] = water_potential_per_year
   
   pot2[count_y,count_x] = pot2[count_y,count_x] + road_potential_per_water * np.minimum(water_flow_required,water_potential_per_year)
      
   minimum_cost_3 = 0.0214764160000129*road_slope_applied**4 - 1.02397109333378*road_slope_applied**3 + 18.4185244000043*road_slope_applied**2 - 151.552444666692*road_slope_applied + 520.40174000009900000000  
   road_slope_applied_3 = road_slope_applied
   theorétical_road_slope_3 = mínimum_theorétical_road_slope

  if topography[count_y,count_x] < topography[count_y,count_x-1]:
   altitude_difference = ((topography[count_y,count_x]-topography[count_y,count_x-1])**2)**0.5/1000
   horizontal_distance = diagonal_road_distance
   mínimum_theorétical_road_slope = altitude_difference/horizontal_distance*100
   road_slope_applied = 0.00000000009479256881*mínimum_theorétical_road_slope**6 - 0.00000006164989263188*mínimum_theorétical_road_slope**5 + 0.00000973488567068692*mínimum_theorétical_road_slope**4 - 0.00053230155937455900*mínimum_theorétical_road_slope**3 + 0.00174393702491216*mínimum_theorétical_road_slope**2 + 0.682497950761899*mínimum_theorétical_road_slope + 0.0329663890424854   
   
   # Road potential in GWh per year
   generation_per_truck_hour = (5.98074425279631*road_slope_applied**2 + 10.3217155154475*road_slope_applied - 5.77372144824585)/24/365/1000
   road_min = np.minimum(roads[count_y+360,count_x],roads[count_y+360,count_x-1])
   trucks_per_hour = 17.8571428571429*road_min**2 + 146.428571428572*road_min + 0.00000000000181898940
   road_potential_per_year = generation_per_truck_hour * trucks_per_hour * 24 * 365
   road_lengh_correction = -0.00000000000870246189*(altitude_difference*1000)**3 + 0.00000014815348746280*(altitude_difference*1000)**2 - 0.00007450553353344210*(altitude_difference*1000) + 1.43839360020761000000
   road_potential_map[count_y,count_x] = road_potential_map[count_y,count_x] + road_potential_per_year * road_lengh_correction
   water_flow_required = trucks_per_hour*33.1* 24 * 365
   water_road_potential_map [count_y,count_x] = water_road_potential_map [count_y,count_x] + water_flow_required    
   road_potential_per_water = road_potential_per_year * road_lengh_correction / water_flow_required

   # Water potential in GWh per year
   water_potential_per_year = rain[int((count_y+360)*3600/4320),int(count_x*3600/4320)]*percent_of_water_extracted*straight_vertical_road_distance*straight_horizontal_road_distance*1000*1000*1000
   water_potential_map[count_y,count_x] = water_potential_per_year
   
   pot2[count_y,count_x] = pot2[count_y,count_x] + road_potential_per_water * np.minimum(water_flow_required,water_potential_per_year)

   minimum_cost_4 = 0.0214764160000129*road_slope_applied**4 - 1.02397109333378*road_slope_applied**3 + 18.4185244000043*road_slope_applied**2 - 151.552444666692*road_slope_applied + 520.40174000009900000000   
   road_slope_applied_4 = road_slope_applied
   theorétical_road_slope_4 = mínimum_theorétical_road_slope

  if topography[count_y,count_x] < topography[count_y,count_x+1]:
   altitude_difference = ((topography[count_y,count_x]-topography[count_y,count_x+1])**2)**0.5/1000
   horizontal_distance = diagonal_road_distance
   mínimum_theorétical_road_slope = altitude_difference/horizontal_distance*100
   road_slope_applied = 0.00000000009479256881*mínimum_theorétical_road_slope**6 - 0.00000006164989263188*mínimum_theorétical_road_slope**5 + 0.00000973488567068692*mínimum_theorétical_road_slope**4 - 0.00053230155937455900*mínimum_theorétical_road_slope**3 + 0.00174393702491216*mínimum_theorétical_road_slope**2 + 0.682497950761899*mínimum_theorétical_road_slope + 0.0329663890424854   
   
   # Road potential in GWh per year
   generation_per_truck_hour = (5.98074425279631*road_slope_applied**2 + 10.3217155154475*road_slope_applied - 5.77372144824585)/24/365/1000
   road_min = np.minimum(roads[count_y+360,count_x],roads[count_y+360,count_x+1])
   trucks_per_hour = 17.8571428571429*road_min**2 + 146.428571428572*road_min + 0.00000000000181898940
   road_potential_per_year = generation_per_truck_hour * trucks_per_hour * 24 * 365
   road_lengh_correction = -0.00000000000870246189*(altitude_difference*1000)**3 + 0.00000014815348746280*(altitude_difference*1000)**2 - 0.00007450553353344210*(altitude_difference*1000) + 1.43839360020761000000
   road_potential_map[count_y,count_x] = road_potential_map[count_y,count_x] + road_potential_per_year * road_lengh_correction
   water_flow_required = trucks_per_hour*33.1* 24 * 365
   water_road_potential_map [count_y,count_x] = water_road_potential_map [count_y,count_x] + water_flow_required    
   road_potential_per_water = road_potential_per_year * road_lengh_correction / water_flow_required

   # Water potential in GWh per year
   water_potential_per_year = rain[int((count_y+360)*3600/4320),int(count_x*3600/4320)]*percent_of_water_extracted*straight_vertical_road_distance*straight_horizontal_road_distance*1000*1000*1000
   water_potential_map[count_y,count_x] = water_potential_per_year
   
   pot2[count_y,count_x] = pot2[count_y,count_x] + road_potential_per_water * np.minimum(water_flow_required,water_potential_per_year)

   minimum_cost_5 = 0.0214764160000129*road_slope_applied**4 - 1.02397109333378*road_slope_applied**3 + 18.4185244000043*road_slope_applied**2 - 151.552444666692*road_slope_applied + 520.40174000009900000000         
   road_slope_applied_5 = road_slope_applied
   theorétical_road_slope_5 = mínimum_theorétical_road_slope

  if topography[count_y,count_x] < topography[count_y+1,count_x-1]:
   altitude_difference = ((topography[count_y,count_x]-topography[count_y+1,count_x-1])**2)**0.5/1000
   horizontal_distance = diagonal_road_distance
   mínimum_theorétical_road_slope = altitude_difference/horizontal_distance*100
   road_slope_applied = 0.00000000009479256881*mínimum_theorétical_road_slope**6 - 0.00000006164989263188*mínimum_theorétical_road_slope**5 + 0.00000973488567068692*mínimum_theorétical_road_slope**4 - 0.00053230155937455900*mínimum_theorétical_road_slope**3 + 0.00174393702491216*mínimum_theorétical_road_slope**2 + 0.682497950761899*mínimum_theorétical_road_slope + 0.0329663890424854   
   
   # Road potential in GWh per year
   generation_per_truck_hour = (5.98074425279631*road_slope_applied**2 + 10.3217155154475*road_slope_applied - 5.77372144824585)/24/365/1000
   road_min = np.minimum(roads[count_y+360,count_x],roads[count_y+360+1,count_x-1])
   trucks_per_hour = 17.8571428571429*road_min**2 + 146.428571428572*road_min + 0.00000000000181898940
   road_potential_per_year = generation_per_truck_hour * trucks_per_hour * 24 * 365
   road_lengh_correction = -0.00000000000870246189*(altitude_difference*1000)**3 + 0.00000014815348746280*(altitude_difference*1000)**2 - 0.00007450553353344210*(altitude_difference*1000) + 1.43839360020761000000
   road_potential_map[count_y,count_x] = road_potential_map[count_y,count_x] + road_potential_per_year * road_lengh_correction
   water_flow_required = trucks_per_hour*33.1* 24 * 365
   water_road_potential_map [count_y,count_x] = water_road_potential_map [count_y,count_x] + water_flow_required    
   road_potential_per_water = road_potential_per_year * road_lengh_correction / water_flow_required

   # Water potential in GWh per year
   water_potential_per_year = rain[int((count_y+360)*3600/4320),int(count_x*3600/4320)]*percent_of_water_extracted*straight_vertical_road_distance*straight_horizontal_road_distance*1000*1000*1000
   water_potential_map[count_y,count_x] = water_potential_per_year
   
   pot2[count_y,count_x] = pot2[count_y,count_x] + road_potential_per_water * np.minimum(water_flow_required,water_potential_per_year)
      
   minimum_cost_6 = 0.0214764160000129*road_slope_applied**4 - 1.02397109333378*road_slope_applied**3 + 18.4185244000043*road_slope_applied**2 - 151.552444666692*road_slope_applied + 520.40174000009900000000      
   road_slope_applied_6 = road_slope_applied
   theorétical_road_slope_6 = mínimum_theorétical_road_slope

  if topography[count_y,count_x] < topography[count_y+1,count_x]:
   altitude_difference = ((topography[count_y,count_x]-topography[count_y-1,count_x-1])**2)**0.5/1000
   horizontal_distance = diagonal_road_distance
   mínimum_theorétical_road_slope = altitude_difference/horizontal_distance*100
   road_slope_applied = 0.00000000009479256881*mínimum_theorétical_road_slope**6 - 0.00000006164989263188*mínimum_theorétical_road_slope**5 + 0.00000973488567068692*mínimum_theorétical_road_slope**4 - 0.00053230155937455900*mínimum_theorétical_road_slope**3 + 0.00174393702491216*mínimum_theorétical_road_slope**2 + 0.682497950761899*mínimum_theorétical_road_slope + 0.0329663890424854   
   
   # Road potential in GWh per year
   generation_per_truck_hour = (5.98074425279631*road_slope_applied**2 + 10.3217155154475*road_slope_applied - 5.77372144824585)/24/365/1000
   road_min = np.minimum(roads[count_y+360,count_x],roads[count_y+360-1,count_x-1])
   trucks_per_hour = 17.8571428571429*road_min**2 + 146.428571428572*road_min + 0.00000000000181898940
   road_potential_per_year = generation_per_truck_hour * trucks_per_hour * 24 * 365
   road_lengh_correction = -0.00000000000870246189*(altitude_difference*1000)**3 + 0.00000014815348746280*(altitude_difference*1000)**2 - 0.00007450553353344210*(altitude_difference*1000) + 1.43839360020761000000
   road_potential_map[count_y,count_x] = road_potential_map[count_y,count_x] + road_potential_per_year * road_lengh_correction
   water_flow_required = trucks_per_hour*33.1* 24 * 365
   water_road_potential_map [count_y,count_x] = water_road_potential_map [count_y,count_x] + water_flow_required    
   road_potential_per_water = road_potential_per_year * road_lengh_correction / water_flow_required

   # Water potential in GWh per year
   water_potential_per_year = rain[int((count_y+360)*3600/4320),int(count_x*3600/4320)]*percent_of_water_extracted*straight_vertical_road_distance*straight_horizontal_road_distance*1000*1000*1000
   water_potential_map[count_y,count_x] = water_potential_per_year
   
   pot2[count_y,count_x] = pot2[count_y,count_x] + road_potential_per_water * np.minimum(water_flow_required,water_potential_per_year)
      
   minimum_cost_7 = 0.0214764160000129*road_slope_applied**4 - 1.02397109333378*road_slope_applied**3 + 18.4185244000043*road_slope_applied**2 - 151.552444666692*road_slope_applied + 520.40174000009900000000       
   road_slope_applied_7 = road_slope_applied
   theorétical_road_slope_7 = mínimum_theorétical_road_slope

  if topography[count_y,count_x] < topography[count_y+1,count_x+1]:
   altitude_difference = ((topography[count_y,count_x]-topography[count_y+1,count_x+1])**2)**0.5/1000
   horizontal_distance = diagonal_road_distance
   mínimum_theorétical_road_slope = altitude_difference/horizontal_distance*100
   road_slope_applied = 0.00000000009479256881*mínimum_theorétical_road_slope**6 - 0.00000006164989263188*mínimum_theorétical_road_slope**5 + 0.00000973488567068692*mínimum_theorétical_road_slope**4 - 0.00053230155937455900*mínimum_theorétical_road_slope**3 + 0.00174393702491216*mínimum_theorétical_road_slope**2 + 0.682497950761899*mínimum_theorétical_road_slope + 0.0329663890424854   
   
   # Road potential in GWh per year
   generation_per_truck_hour = (5.98074425279631*road_slope_applied**2 + 10.3217155154475*road_slope_applied - 5.77372144824585)/24/365/1000
   road_min = np.minimum(roads[count_y+360,count_x],roads[count_y+360+1,count_x+1])
   trucks_per_hour = 17.8571428571429*road_min**2 + 146.428571428572*road_min + 0.00000000000181898940
   road_potential_per_year = generation_per_truck_hour * trucks_per_hour * 24 * 365
   road_lengh_correction = -0.00000000000870246189*(altitude_difference*1000)**3 + 0.00000014815348746280*(altitude_difference*1000)**2 - 0.00007450553353344210*(altitude_difference*1000) + 1.43839360020761000000
   road_potential_map[count_y,count_x] = road_potential_map[count_y,count_x] + road_potential_per_year * road_lengh_correction
   water_flow_required = trucks_per_hour*33.1* 24 * 365
   water_road_potential_map [count_y,count_x] = water_road_potential_map [count_y,count_x] + water_flow_required    
   road_potential_per_water = road_potential_per_year * road_lengh_correction / water_flow_required

   # Water potential in GWh per year
   water_potential_per_year = rain[int((count_y+360)*3600/4320),int(count_x*3600/4320)]*percent_of_water_extracted*straight_vertical_road_distance*straight_horizontal_road_distance*1000*1000*1000
   water_potential_map[count_y,count_x] = water_potential_per_year
   
   pot2[count_y,count_x] = pot2[count_y,count_x] + road_potential_per_water * np.minimum(water_flow_required,water_potential_per_year)
      
   minimum_cost_8 = 0.0214764160000129*road_slope_applied**4 - 1.02397109333378*road_slope_applied**3 + 18.4185244000043*road_slope_applied**2 - 151.552444666692*road_slope_applied + 520.40174000009900000000       
   road_slope_applied_8 = road_slope_applied
   theorétical_road_slope_8 = mínimum_theorétical_road_slope
    
 # print("xxxxx",road_slope_applied_1,road_slope_applied_2,road_slope_applied_3,road_slope_applied_4,road_slope_applied_5,road_slope_applied_6,road_slope_applied_7,road_slope_applied_8)
  minimum_cost[count_y,count_x] = min([minimum_cost_1,minimum_cost_2,minimum_cost_3,minimum_cost_4,minimum_cost_5,minimum_cost_6,minimum_cost_7,minimum_cost_8])
  road_slope[count_y,count_x] = max([road_slope_applied_1,road_slope_applied_2,road_slope_applied_3,road_slope_applied_4,road_slope_applied_5,road_slope_applied_6,road_slope_applied_7,road_slope_applied_8])
  theorétical_road_slope[count_y,count_x] = max([theorétical_road_slope_1,theorétical_road_slope_2,theorétical_road_slope_3,theorétical_road_slope_4,theorétical_road_slope_5,theorétical_road_slope_6,theorétical_road_slope_7,theorétical_road_slope_8])

                 #   flooded_area = np.array([[0,0]]) # This array stores the coordinates of the flooded area
                 #   flooded_area= np.delete(flooded_area, 0, 0)   
                 #   flooded_area = np.vstack((flooded_area, [[y,x]]))
                    
  count_x = count_x +1
  
 count_y = count_y +1
 #print(count_y) 
#plt2.imshow(pot2)
#plt2.imshow(minimum_cost)


#%%

pot_low2 = np.zeros(shape=(120,360))
total_potential = 0

count_y = 0
while count_y<120:
 count_x = 0
 while count_x<360:
   if np.sum(pot2[count_y*12:count_y*12+12,count_x*12:count_x*12+12]) < 0:
      pot_low2[count_y,count_x] = np.nan
   elif np.sum(pot2[count_y*12:count_y*12+12,count_x*12:count_x*12+12]) == 0:
      pot_low2[count_y,count_x] = np.nan
   elif math.log(np.sum(pot2[count_y*12:count_y*12+12,count_x*12:count_x*12+12]),10) <= 0.1:
      pot_low2[count_y,count_x] = np.nan
   else:
      #pot_low2[count_y,count_x] = np.sum(pot2[count_y*12:count_y*12+12,count_x*12:count_x*12+12])
      pot_low2[count_y,count_x] = math.log(np.sum(pot2[count_y*12:count_y*12+12,count_x*12:count_x*12+12]),10)
      total_potential = total_potential + np.sum(pot2[count_y*12:count_y*12+12,count_x*12:count_x*12+12])
   count_x = count_x +1
 count_y = count_y +1

plt2.imshow(pot_low2)



#%%

road_potential_map2 = np.zeros(shape=(120,360))

count_y = 0
while count_y<120:
 count_x = 0
 while count_x<360:
   if np.amax(road_potential_map[count_y*12:count_y*12+12,count_x*12:count_x*12+12]) < 0:
      road_potential_map2[count_y,count_x] = np.nan
   elif np.amax(road_potential_map[count_y*12:count_y*12+12,count_x*12:count_x*12+12]) == 0:
      road_potential_map2[count_y,count_x] = np.nan
   elif np.sum(road_potential_map[count_y*12:count_y*12+12,count_x*12:count_x*12+12]) <= 0.001:
      road_potential_map2[count_y,count_x] = np.nan
   else:
      road_potential_map2[count_y,count_x] = np.sum(road_potential_map[count_y*12:count_y*12+12,count_x*12:count_x*12+12])
      #road_potential_map2[count_y,count_x] = math.log(np.sum(pot2[count_y*12:count_y*12+12,count_x*12:count_x*12+12]),10)
   count_x = count_x +1
 count_y = count_y +1

plt2.imshow(road_potential_map2)


#%%

water_potential_map2 = np.zeros(shape=(120,360))

count_y = 0
while count_y<120:
 count_x = 0
 while count_x<360:
   if np.amax(water_potential_map[count_y*12:count_y*12+12,count_x*12:count_x*12+12]) < 0:
      water_potential_map2[count_y,count_x] = np.nan
   elif np.amax(water_potential_map[count_y*12:count_y*12+12,count_x*12:count_x*12+12]) == 0:
      water_potential_map2[count_y,count_x] = np.nan
   elif np.sum(water_potential_map[count_y*12:count_y*12+12,count_x*12:count_x*12+12]) <= 0.001:
      water_potential_map2[count_y,count_x] = np.nan
   else:
      water_potential_map2[count_y,count_x] = np.sum(water_potential_map[count_y*12:count_y*12+12,count_x*12:count_x*12+12])
      #road_potential_map2[count_y,count_x] = math.log(np.sum(pot2[count_y*12:count_y*12+12,count_x*12:count_x*12+12]),10)
   count_x = count_x +1
 count_y = count_y +1

plt2.imshow(water_potential_map2)
   

#%%

water_road_potential_map2 = np.zeros(shape=(120,360))

count_y = 0
while count_y<120:
 count_x = 0
 while count_x<360:
   if np.amax(water_road_potential_map[count_y*12:count_y*12+12,count_x*12:count_x*12+12]) < 0:
      water_road_potential_map2[count_y,count_x] = np.nan
   elif np.amax(water_road_potential_map[count_y*12:count_y*12+12,count_x*12:count_x*12+12]) == 0:
      water_road_potential_map2[count_y,count_x] = np.nan
   elif np.sum(water_road_potential_map[count_y*12:count_y*12+12,count_x*12:count_x*12+12]) <= 0.001:
      water_road_potential_map2[count_y,count_x] = np.nan
   else:
      water_road_potential_map2[count_y,count_x] = np.sum(water_road_potential_map[count_y*12:count_y*12+12,count_x*12:count_x*12+12])
      #water_road_potential_map2[count_y,count_x] = math.log(np.sum(water_road_potential_map[count_y*12:count_y*12+12,count_x*12:count_x*12+12]),10)
   count_x = count_x +1
 count_y = count_y +1

plt2.imshow(water_road_potential_map2)
   
#%%

minimum_cost2 = np.zeros(shape=(120,360))

count_y = 0
while count_y<120:
 count_x = 0
 while count_x<360:
   if np.amax(np.amin(minimum_cost[count_y*12:count_y*12+12,count_x*12:count_x*12+12])) > 199.5:
      minimum_cost2[count_y,count_x] = np.nan
   elif np.amax(np.amin(minimum_cost[count_y*12:count_y*12+12,count_x*12:count_x*12+12])) == 199.5:
      minimum_cost2[count_y,count_x] = np.nan
   elif np.amax(np.amin(minimum_cost[count_y*12:count_y*12+12,count_x*12:count_x*12+12])) == 0:
      minimum_cost2[count_y,count_x] = np.nan
   else:
      #pot_low2[count_y,count_x] = np.sum(pot2[count_y*12:count_y*12+12,count_x*12:count_x*12+12])
      minimum_cost2[count_y,count_x] = np.amin(minimum_cost[count_y*12:count_y*12+12,count_x*12:count_x*12+12])
   count_x = count_x +1
 count_y = count_y +1

plt2.imshow(minimum_cost2)



#%%

road_slope2 = np.zeros(shape=(120,360))

count_y = 0
while count_y<120:
 count_x = 0
 while count_x<360:
   if np.amax(np.amax(road_slope[count_y*12:count_y*12+12,count_x*12:count_x*12+12])) < 0.005:
      road_slope2[count_y,count_x] = np.nan
   elif np.amax(np.amax(road_slope[count_y*12:count_y*12+12,count_x*12:count_x*12+12])) == 0:
      road_slope2[count_y,count_x] = np.nan
#   elif np.amax(np.amax(road_slope[count_y*12:count_y*12+12,count_x*12:count_x*12+12])) > 1:
#      road_slope2[count_y,count_x] = np.nan
   else:
      #pot_low2[count_y,count_x] = np.sum(pot2[count_y*12:count_y*12+12,count_x*12:count_x*12+12])
      road_slope2[count_y,count_x] = np.amax(road_slope[count_y*12:count_y*12+12,count_x*12:count_x*12+12])
   count_x = count_x +1
 count_y = count_y +1

plt2.imshow(road_slope2)



#%%

theoretical2 = np.zeros(shape=(120,360))

count_y = 0
while count_y<120:
 count_x = 0
 while count_x<360:
   if np.amax(np.amax(theorétical_road_slope[count_y*12:count_y*12+12,count_x*12:count_x*12+12])) < 0.005:
      theoretical2[count_y,count_x] = np.nan
   elif np.amax(np.amax(theorétical_road_slope[count_y*12:count_y*12+12,count_x*12:count_x*12+12])) == 0:
      theoretical2[count_y,count_x] = np.nan
#   elif np.amax(np.amax(theoretical[count_y*12:count_y*12+12,count_x*12:count_x*12+12])) > 1:
#      theoretical2[count_y,count_x] = np.nan
   else:
      #pot_low2[count_y,count_x] = np.sum(pot2[count_y*12:count_y*12+12,count_x*12:count_x*12+12])
      theoretical2[count_y,count_x] = np.amax(theorétical_road_slope[count_y*12:count_y*12+12,count_x*12:count_x*12+12])
   count_x = count_x +1
 count_y = count_y +1

plt2.imshow(theoretical2)



#%%

class FixPointNormalize(matplotlib.colors.Normalize):
    """ 
    Inspired by https://stackoverflow.com/questions/20144529/shifted-colorbar-matplotlib
    Subclassing Normalize to obtain a colormap with a fixpoint 
    somewhere in the middle of the colormap.

    This may be useful for a `terrain` map, to set the "sea level" 
    to a color in the blue/turquise range. 
    """
    def __init__(self, vmin=None, vmax=None, sealevel=0, col_val = 0.21875, clip=False):
        # sealevel is the fix point of the colormap (in data units)
        self.sealevel = sealevel
        # col_val is the color value in the range [0,1] that should represent the sealevel.
        self.col_val = col_val
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.sealevel, self.vmax], [0, self.col_val, 1]
        return np.ma.masked_array(np.interp(value, x, y))


tds4 = xr.Dataset()
#tds4['data'] = (['Latitude','Longuitude'],corrected_20_degrees )
#tds4['data'] = (['Latitude','Longuitude'],corrected_22_degrees )
#tds4['data'] = (['Latitude','Longuitude'],corrected_24_degrees )
#tds4['data'] = (['Latitude','Longuitude'],corrected_26_degrees )
#tds4['data'] = (['Latitude','Longuitude'],corrected_28_degrees )
#tds4['data'] = (['Latitude','Longuitude'],corrected_30_degrees )
tds4['data'] = (['Latitude','Longuitude'],pot_low2)#roads_low)#rain_low)#pot_low)



tds4['Latitude'] = np.arange(60,-60,-0.083335*12)
tds4['Longuitude'] = np.arange(-180,180,0.08335*12)

# Plot bathymetry_map_with_projects


cmap = 'jet_r'#'Greys'#'terrain'#'jet'#'jet_r'
#norm = FixPointNormalize(sealevel=-1)#, vmax=20000)
fig = plt.figure()
ax = plt.axes(projection= ccrs.PlateCarree())
ax.set_global();
ax.add_feature(cartopy.feature.LAND, zorder=0)
ax.add_feature(cartopy.feature.OCEAN, facecolor='#d3e9ed', zorder=0)        
ax.coastlines(linewidth=0.3, edgecolor='k')#'#778899'
ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='#778899')

p = tds4.data.plot(cmap=cmap)#, norm=norm) ### plot using xarray
ax.set_extent([-180, 180, -60, 60], ccrs.PlateCarree())

#ax.set_xticks(np.arange(-90, 0, 30))
#ax.set_yticks(np.arange(45,-45 ,-30))

lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)


#%%

class FixPointNormalize(matplotlib.colors.Normalize):
    """ 
    Inspired by https://stackoverflow.com/questions/20144529/shifted-colorbar-matplotlib
    Subclassing Normalize to obtain a colormap with a fixpoint 
    somewhere in the middle of the colormap.

    This may be useful for a `terrain` map, to set the "sea level" 
    to a color in the blue/turquise range. 
    """
    def __init__(self, vmin=None, vmax=None, sealevel=0, col_val = 0.21875, clip=False):
        # sealevel is the fix point of the colormap (in data units)
        self.sealevel = sealevel
        # col_val is the color value in the range [0,1] that should represent the sealevel.
        self.col_val = col_val
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.sealevel, self.vmax], [0, self.col_val, 1]
        return np.ma.masked_array(np.interp(value, x, y))


tds4 = xr.Dataset()
#tds4['data'] = (['Latitude','Longuitude'],corrected_20_degrees )
#tds4['data'] = (['Latitude','Longuitude'],corrected_22_degrees )
#tds4['data'] = (['Latitude','Longuitude'],corrected_24_degrees )
#tds4['data'] = (['Latitude','Longuitude'],corrected_26_degrees )
#tds4['data'] = (['Latitude','Longuitude'],corrected_28_degrees )
#tds4['data'] = (['Latitude','Longuitude'],corrected_30_degrees )
tds4['data'] = (['Latitude','Longuitude'],road_slope)#rain2) #top)

#1440,4320))1801,3600     2160,4320))

tds4['Latitude'] = np.arange(60,-60,-0.083335)#*2160/1801)
tds4['Longuitude'] = np.arange(-180,180,0.08335)#*4320/3600)

# Plot bathymetry_map_with_projects

cmap = 'Blues'#'viridis'#'Greys'#'jet'#'Greys'#'Blues'#'jet'#'viridis'#'Greys'#'terrain'#'jet'
norm = FixPointNormalize( vmax=2)
fig = plt.figure()
ax = plt.axes(projection= ccrs.PlateCarree())
ax.set_global();
ax.add_feature(cartopy.feature.LAND, zorder=0)
ax.add_feature(cartopy.feature.OCEAN, facecolor='#d3e9ed', zorder=0)        
ax.coastlines(linewidth=0.3, edgecolor='k')#'#778899'
ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='#778899')

p = tds4.data.plot(cmap=cmap,norm=norm) ### plot using xarray
ax.set_extent([-180, 180, -60, 60], ccrs.PlateCarree())

#ax.set_xticks(np.arange(-90, 0, 30))
#ax.set_yticks(np.arange(45,-45 ,-30))

lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)

#%%

pot_low2 = np.zeros(shape=(120,360))
total_potential = 0

count_y = 0
while count_y<120:
 count_x = 0
 while count_x<360:
   if np.amax(pot2[count_y*12:count_y*12+12,count_x*12:count_x*12+12]) < 0:
      pot_low2[count_y,count_x] = np.nan
   elif np.amax(pot2[count_y*12:count_y*12+12,count_x*12:count_x*12+12]) == 0:
      pot_low2[count_y,count_x] = np.nan
   elif np.sum(pot2[count_y*12:count_y*12+12,count_x*12:count_x*12+12]) <= 0.001:
      pot_low2[count_y,count_x] = np.nan
   else:
      pot_low2[count_y,count_x] = np.sum(pot2[count_y*12:count_y*12+12,count_x*12:count_x*12+12])
      #pot_low2[count_y,count_x] = math.log(np.sum(pot2[count_y*12:count_y*12+12,count_x*12:count_x*12+12]),10)
      total_potential = total_potential + np.sum(pot2[count_y*12:count_y*12+12,count_x*12:count_x*12+12])
   count_x = count_x +1
 count_y = count_y +1

plt2.imshow(pot_low2)


#%%

minimum_cost2 = np.zeros(shape=(120,360))

count_y = 0
while count_y<120:
 count_x = 0
 while count_x<360:
   if np.amax(np.amin(minimum_cost[count_y*12:count_y*12+12,count_x*12:count_x*12+12])) > 100:
      minimum_cost2[count_y,count_x] = np.nan
   elif np.amax(np.amin(minimum_cost[count_y*12:count_y*12+12,count_x*12:count_x*12+12])) == 100:
      minimum_cost2[count_y,count_x] = np.nan
   elif np.amax(np.amin(minimum_cost[count_y*12:count_y*12+12,count_x*12:count_x*12+12])) == 0:
      minimum_cost2[count_y,count_x] = np.nan
   else:
      #pot_low2[count_y,count_x] = np.sum(pot2[count_y*12:count_y*12+12,count_x*12:count_x*12+12])
      minimum_cost2[count_y,count_x] = np.amin(minimum_cost[count_y*12:count_y*12+12,count_x*12:count_x*12+12])
   count_x = count_x +1
 count_y = count_y +1

plt2.imshow(minimum_cost2)



#%%
#cost_curve = np.array([[0,0]]) # This array stores the coordinates of the flooded area
#cost_curve = np.delete(cost_curve, 0, 0)   
#cost_curve = np.vstack((cost_curve, [[pot2[count_y,count_x],minimum_cost[count_y,count_x]]]))   


cost_curve = np.zeros(shape=(26547*2,3))


count_z = 0
count_y = 0
while count_y<1440:
 count_x = 0
 while count_x<4320:
   if minimum_cost[count_y,count_x] <= 100:  
    cost_curve[count_z,0] = pot2[count_y,count_x]
    cost_curve[count_z,1] = minimum_cost[count_y,count_x]   
    cost_curve[count_z,2] = continents[count_y,count_x]
    count_z = count_z +1    
    cost_curve[count_z,0] = pot2[count_y,count_x]
    cost_curve[count_z,1] = minimum_cost[count_y,count_x]   
    cost_curve[count_z,2] = continents[count_y,count_x] 
    count_z = count_z +1
   count_x = count_x +1
 count_y = count_y +1
 print(count_y)
 
df = pd.DataFrame (cost_curve)

filepath = 'C:/Users/julia/Documents/Julian/IIASA/Electric Truck Hydropower/cost curve (continents).xlsx'

df.to_excel(filepath, index=False)

cost_curve = np.zeros(shape=(26547*2,2))

#%%

continents = np.zeros(shape=(1440,4320))

# 1 North America
continents[0:325,0:1570] = 1
continents[0:600,0:600] = 1
# 3 South America
continents[570:1440,601:1900] = 3
# 2 Central America
continents[326:570,601:1600] = 2
continents[571:650,601:1230] = 2
# 5 Africa
continents[270:1440,1901:2900] = 5
# 7 Oceania
continents[840:1440,3400:4320] = 7
# 4 Europe
continents[0:270,1601:2500] = 4
# 6 Asia
continents[0:320,2500:4320] = 6
continents[320:390,2570:4320] = 6
continents[390:410,10+2570:4320] = 6
continents[410:430,10+2581:4320] = 6
continents[430:450,10+2591:4320] = 6
continents[450:470,10+2601:4320] = 6
continents[470:490,10+2611:4320] = 6
continents[490:510,10+2621:4320] = 6
continents[510:530,10+2631:4320] = 6
continents[530:550,10+2641:4320] = 6
continents[550:570,10+2651:4320] = 6
continents[570:590,10+2661:4320] = 6
continents[0:840,2901:4320] = 6

plt2.imshow(continents)

top = np.zeros(shape=(1440,4320))

 
count_y = 0
while count_y<1440:
 count_x = 0
 while count_x<4320:
   top[count_y,count_x] = topography[count_y,count_x]+continents[count_y,count_x]*200
   count_x = count_x +1
 count_y = count_y +1
 

plt2.imshow(top)

#plt2.imshow(topography)