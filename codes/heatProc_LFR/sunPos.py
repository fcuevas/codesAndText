# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 19:51:06 2015

@author: fcuevas
"""



from numpy import cos, sin, tan, arccos, arctan, deg2rad, rad2deg, arcsin
import numpy as np
#import panda as pn

def sunPos(lat,day,hour,minute):
    decl = 23.45 * sin(deg2rad (360*(284+day)/365))
    w=-(12-(hour+minute/60.))*15
    cos_theta_zeta = cos(deg2rad(lat))*cos(deg2rad(decl))*cos(deg2rad(w))+sin(deg2rad(lat))*sin(deg2rad(decl))
    theta_zeta = rad2deg(arccos(cos_theta_zeta))
            
    gamma_s = np.sign(w)*np.abs(rad2deg(arccos((cos(deg2rad(theta_zeta))*sin(deg2rad(lat))-sin(deg2rad(decl)))/(sin(deg2rad(theta_zeta))*cos(deg2rad(lat))))))

            
    return theta_zeta, gamma_s
    
    
#def G_on(day):
#    G_sc = 1367.0
#    G_on = G_sc * (1+0.033*cos(deg2rad(360*day/365)))
#    return G_on
    
fileDebug = False


if fileDebug == True:


    # Resolution of the results. If it is True, creates a file called "hourlyResult.dat" and prints on it the results per hour
    hourlyRes=True
    
    # Resolution of the results. If it is True, creates a file called "stepResult.dat" and prints on it the results in the time step defined 
    # in the variable step
    stepRes=False
    
    #Variable to print the results to a file or not
    printRes=True
    
    #Total days to be used in the calculation (365 -> 1 year)
    days=365
    
    #definition of step in calculation (minutes)
    step = 10
    
    #Heigth of the absorber center (m)
    H_col=7
    
    #Lenght of the collector (m)
    l_col=100
    
    # Solar Constant, radiation outside the atmosphere (W/m2)
    G_sc = 1367 
    
    #Latitude of the place to analize (+ north hemisphere, - south hemisphere)
    # Latitude goes from -90 to 90
    latitude = -33.0
    
    #Longitude of the place to analize
    longitude = 89.4
    
    L_st = 90
    
    min_frac = 60 / step
    
    hour_year = 0
    
    if hourlyRes:
        r = open("hourlyResult.dat", "w")
        headerHR = "{0:^12} {1:^12} {2:^12} {3:^12} {4:^12} {5:^12}\
    {6:^12} {7:^12} {8:^12} {9:^12} {10:^12} {11:^12}\
    {12:^12} {13:^12} {14:12} {15:12}"\
    .format("Hour_year", "Day",  "Hour",  "G_on", "G_o", "decl", "w_sunset",\
    "w_sunrise", "w", "Light_hours", "Theta_zeta", "Gamma_s",\
    "Theta_t", "Theta_l", "Theta_i", "f_end         \n")
        r.write(headerHR)
                    
    if stepRes:
        s = open("stepResult.dat", "w")
        headerSR = "{0:10} {1:10} {2:10} {3:10} {4:10} {5:10}\
                    {6:10} {7:10} {8:10} {9:10} {10:10} {11:10}\
                    {12:10} {13:10} {14:10} {15:10}"\
                    .format("Day", "Hour", "Minute", "G_on", "G_o", "decl", "w_sunset",\
                    "w_sunrise", "w", "Light_hours", "Theta_zeta", "Gamma_s",\
                    "Theta_t", "Theta_l", "Theta_i", "f_end\n")
        s.write(headerSR)
        
        
    gamma = 90
    
    G_onL = []
    B_L = []
    E_L = []
    decl_L = []
    
    
    
    for day in range(1,days+1):
        
        G_on = G_sc * (1+0.033*cos(deg2rad(360*day/365)))
        G_onL.append(G_on)
        
        B = (day-1) * 360/365
        B_L.append(B)
        
        E = 292.2 * (0.000075 + 0.001868 * cos(B) - 0.032077 * sin(deg2rad(B))\
            - 0.014615 * cos(deg2rad(2*B)) - 0.04089 * sin(deg2rad(2*B)))
        E_L.append(E)
            
        decl = 23.45 * sin(deg2rad (360*(284+day)/365))
        decl_L.append(decl)
        
        cos_omega_sunset = -tan(deg2rad(latitude)) * tan(deg2rad(decl)) 
        omega_sunset = rad2deg(arccos(cos_omega_sunset))
        N=omega_sunset*2/15
        omega_sunrise = -omega_sunset
        H_o = 24*3600*G_sc/np.pi*G_on*(cos(deg2rad(latitude))*cos(deg2rad(decl))*cos(deg2rad(omega_sunset))\
              +np.pi*omega_sunset/180*sin(deg2rad(latitude))*sin(deg2rad(decl)))
        for hour in range(0,24):
            hour_year = hour_year+1
            for minute in range(0,min_frac):
                frac_min = step*minute
                hour_frac = hour + frac_min/60
                
                w=-(12-(hour+frac_min/60.))*15
                cos_theta_zeta = cos(deg2rad(latitude))*cos(deg2rad(decl))*cos(deg2rad(w))+sin(deg2rad(latitude))*sin(deg2rad(decl))
                theta_zeta = rad2deg(arccos(cos_theta_zeta))
                
                gamma_s = np.sign(w)*np.abs(rad2deg(arccos((cos(deg2rad(theta_zeta))*sin(deg2rad(latitude))-sin(deg2rad(decl)))/(sin(deg2rad(theta_zeta))*cos(deg2rad(latitude))))))
                
                cos_theta = (cos(deg2rad(theta_zeta))**2 + cos(deg2rad(decl))**2*sin(deg2rad(w))**2)**0.5
                theta = rad2deg(arccos(cos_theta))            
                theta_t=rad2deg(arctan(sin(deg2rad(gamma_s))*tan(deg2rad(theta_zeta))))
                theta_l=rad2deg(arctan(cos(deg2rad(gamma_s))*tan(deg2rad(theta_zeta))))
                theta_i=rad2deg(arcsin(cos(deg2rad(gamma_s))*sin(deg2rad(theta_zeta))))
                tan_beta = tan(deg2rad(theta_zeta))*np.abs(cos(deg2rad(gamma*np.sign(gamma_s)-gamma_s)))
                beta_p = rad2deg(arctan(tan_beta))
                G_o = G_on*cos_theta_zeta
                
                thetaFunc, gammaFunc = sunPos(latitude, day, hour, frac_min)
                
                
                if stepRes and printRes:
    
                    s.write("%2d" % day) 
                    s.write("%6d" % hour)
                    s.write("%6.1f" % frac_min)
                    s.write("%10.1f" % G_on)
                    s.write("%10.1f" % G_o)
                    s.write("%10.1f" % decl)
                    s.write("%10.1f" % omega_sunset)
                    s.write("%10.1f" % omega_sunrise)
                    s.write("{: 10.2f}; {: 10.2f}".format(w))
                    s.write("%10.1f" % N)
                    s.write("%10.1f" % theta_zeta)
                    s.write("%10.1f" % thetaFunc)
                    s.write("%10.1f" % gamma_s)
                    s.write("%10.1f" % gammaFunc)
                    s.write("%10.1f" % theta_t)
                    s.write("%10.1f" % theta_l)
                    s.write("%10.1f" % theta_i)
                
                    s.write("\n")
                
            if hourlyRes and printRes:
                
                r.write("{: 12d}".format(hour_year)) 
                r.write("{: 12d}".format(day))            
                r.write("{: 12d}".format(hour)) 
                r.write("{: 12.1f}".format(G_on))
                r.write("{: 12.1f}".format(G_o))
                r.write("{: 12.1f}".format(decl))
                r.write("{: 12.1f}".format(omega_sunset))
                r.write("{: 12.1f}".format(omega_sunrise))
                r.write("{: 12.1f}".format(w))
                r.write("{: 12.1f}".format(N))
                r.write("{: 12.1f}".format(theta_zeta))
                r.write("{: 12.1f}".format(thetaFunc))
                r.write("{: 12.1f}".format(gamma_s))
                r.write("{: 12.1f}".format(gammaFunc))
                r.write("{: 12.1f}".format(theta_t))
                r.write("{: 12.1f}".format(theta_l))
                r.write("{: 12.1f}".format(theta_i))
    
                r.write("\n")
            
        
    
    r.close()
    s.close()   