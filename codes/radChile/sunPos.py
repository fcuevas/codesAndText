# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 19:51:06 2015

@author: fcuevas
"""


from numpy import cos, sin, tan, arccos, deg2rad, rad2deg
import numpy as np
#import panda as pn

def sunPos(lat,day,hour,minute=0.0):
    decl = declination(day)
    w=-(12-(hour+minute/60.))*15
    cos_theta_zeta = cos(deg2rad(lat))*cos(deg2rad(decl))*cos(deg2rad(w))+sin(deg2rad(lat))*sin(deg2rad(decl))
    theta_zeta = rad2deg(arccos(cos_theta_zeta))
            
    gamma_s = np.sign(w)*np.abs(rad2deg(arccos((cos(deg2rad(theta_zeta))\
    *sin(deg2rad(lat))-sin(deg2rad(decl)))/(sin(deg2rad(theta_zeta))*cos(deg2rad(lat))))))
    
    if np.isnan(gamma_s)==True:
        if lat > 0.0:
            gamma_s= 0.0
        else:
            gamma_s= 180.0
                           
    return theta_zeta, gamma_s
   
    
def G_onSun(day):
    G_sc = 1367.0
    G_onS = G_sc * (1+0.033*cos(deg2rad(360*day/365)))
    
    return G_onS
    
def w_sunset(lat, day):
    decl = declination(day)
    cos_omega_sunset = -tan(deg2rad(lat)) * tan(deg2rad(decl)) 
    omega_sunset = rad2deg(arccos(cos_omega_sunset))
    return omega_sunset
    
def N_day(lat, day):
    omega_sunset = w_sunset(lat, day)
    N=omega_sunset*2/15
    return N
    
def omega(hour, minute=0.0):
    w=-(12-(hour+minute/60.))*15
    return w
    

            
                
def declination(day):
    decl = 23.45 * sin(deg2rad (360*(284+day)/365))
    
    return decl
    
    
def E_t(day):
    B = (day-1) * 360/365       
    E = 292.2 * (0.000075 + 0.001868 * cos(B) - 0.032077 * sin(deg2rad(B))\
            - 0.014615 * cos(deg2rad(2*B)) - 0.04089 * sin(deg2rad(2*B)))
            
    return E

debug = True
if debug:    

    # Resolution of the results. If it is True, creates a file called "hourlyResult.dat" and prints on it the results per hour
    hourlyRes=True
    
    # Resolution of the results. If it is True, creates a file called "stepResult.dat" and prints on it the results in the time step defined 
    # in the variable step
    stepRes=False
    
    
    #Total days to be used in the calculation (365 -> 1 year)
    days=365
    
    #definition of step in calculation (minutes)
    step = 10
    
    #Latitude of the place to analize (+ north hemisphere, - south hemisphere)
    # Latitude goes from -90 to 90
    latitude = -23.0
    
    #Longitude of the place to analize
    longitude = 89.4
    
    L_st = 90
    
    min_frac = 60 / step
    
    hour_year = 0
    
    if hourlyRes:
        r = open("hourlyResult.dat", "w")
        headerHR = "{0:^12} {1:^12} {2:^12} {3:^12} {4:^12} {5:^12}\
    {6:^12} {7:^12} {8:^12} {9:^12}"\
    .format("Hour_year", "Day",  "Hour",  "decl", "w_sunset",\
    "w_sunrise", "w", "Light_hours", "Theta_zeta", "Gamma_s     \n")
        r.write(headerHR)
                    
    if stepRes:
        s = open("stepResult.dat", "w")
        headerSR = "{0:^12} {1:^12} {2:^12} {3:^12} {4:^12} {5:^12}\
    {6:^12} {7:^12} {8:^12} {9:^12}"\
    .format("Day",  "Hour","Minute", "decl", "w_sunset",\
    "w_sunrise", "w", "Light_hours", "Theta_zeta", "Gamma_s     \n")
        s.write(headerSR)
        
        
    gamma = 90
   
    for day in range(1,days+1):
        for hour in range(0,24):
            hour_year = hour_year+1
            
            decl = declination(day)
            omega_sunset = w_sunset(latitude, day)
            N = N_day(latitude, day)
            
            omega_sunrise = -omega_sunset
            w = omega(hour)
            theta_zeta, gamma_s = sunPos(latitude, day, hour) 
            
            
            if stepRes:
            
                for minute in range(0,min_frac):
                    
                    frac_min = step*minute
                    hour_frac = hour + frac_min/60
                    
                    decl = declination(day)
                    omega_sunset = w_sunset(latitude, day)
                    N = N_day(latitude, day)
                    
                    omega_sunrise = -omega_sunset
                    w = omega(hour, minute)
                    theta_zeta, gamma_s = sunPos(latitude, day, hour, frac_min)                
                    
            
                    s.write("{: 12d}".format(day))            
                    s.write("{: 12d}".format(hour)) 
                    s.write("{: 12.1f}".format(frac_min))
                    s.write("{: 12.1f}".format(decl))
                    s.write("{: 12.1f}".format(omega_sunset))
                    s.write("{: 12.1f}".format(omega_sunrise))
                    s.write("{: 12.1f}".format(w))
                    s.write("{: 12.1f}".format(N))
                    s.write("{: 12.1f}".format(theta_zeta))
                    s.write("{: 12.1f}".format(gamma_s))              
                    s.write("\n")
                
            if hourlyRes:
                
                r.write("{: 12d}".format(hour_year)) 
                r.write("{: 12d}".format(day))            
                r.write("{: 12d}".format(hour)) 
                r.write("{: 12.1f}".format(decl))
                r.write("{: 12.1f}".format(omega_sunset))
                r.write("{: 12.1f}".format(omega_sunrise))
                r.write("{: 12.1f}".format(w))
                r.write("{: 12.1f}".format(N))
                r.write("{: 12.1f}".format(theta_zeta))
                r.write("{: 12.1f}".format(gamma_s))
                r.write("\n")
            
        
    if hourlyRes:
        r.close()
        
    if stepRes:
        s.close()   