# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 19:51:06 2015

@author: fcuevas
"""

from numpy import cos, sin, tan, arccos, arcsin, arctan, sqrt, deg2rad, rad2deg
import numpy as np

#import panda as pn

hourlyRes=True
stepRes=True

days=365

#definition of step in calculation (minutes)
step = 30

H_abs = 7.0
truncSec= -0.15
l_col = 100.0
N = 10
W = 0.5
d = 0.15

s = N/2 - N/2.


x_mirror = 0
#c_mirror = np.array([np.zeros(N)])
#print c_mirror

t1 = rad2deg(arctan(-2.5/H_abs))
t2 = rad2deg(arctan(2.5/H_abs))
rho = (30. - t1)/2.
rho2=(30. - t2)/2

print t1, t2, rho, rho2

if s == 0.0:
    d1 = d/2
    x_mirror = np.zeros(N/2)
    for x in range(len(x_mirror)):
        x_mirror[x] = d1 + W/2 + (d + W)*x 
        
    x_mirrorLeft = sorted(x_mirror, reverse=True)    
    c_mirror = np.concatenate((np.negative(x_mirrorLeft), x_mirror))
   
    
else:
    x_mirror = np.zeros(N/2+1)
    x_mirror[0] = 0.0
    for x in range(1, len(x_mirror)):
        x_mirror[x] = W/2 + (d + W/2)*x

    x_mirrorLeft = sorted(x_mirror, reverse=True)    
    x_mirrorLeft.remove(0)
    c_mirror = np.concatenate((np.negative(x_mirrorLeft),x_mirror))
    
        




# W/m2

print x_mirror
print c_mirror

focal_l = sqrt(c_mirror**2 + H_abs**2)
print focal_l

def coordMirror(W, xCent, xLen, beta, foc):

    x1 = (xCent - W/2.)*cos(deg2rad(beta))
    y1 = -l_col/2.
    z1 = (xCent - W/2.)*sin(deg2rad(beta))

    x2 = (xCent + W/2.)*cos(deg2rad(beta))
    y2 = -l_col/2.
    z2 = (xCent + W/2.)*sin(deg2rad(beta))
    c1 = np.array([[x1,y1,z1],[x2,y2,z2], [x2,-y2,z2], [x1,-y1,z1]])
    
    return c1

def inclMirror(theta_t, h_abs, x_m, Dz):
    beta = (theta_t - rad2deg(arctan(x_m/(h_abs + Dz))))/2
    return beta

for mirr in c_mirror:
    m1 = coordMirror(W, c_mirror[mirr], l_col, 0, 1)




print m1


G_sc = 1367 


latitude = 43.0
longitude = 89.4

L_st = 90

#day_year = 365
#hour_day = 24
#min_hour = 60
#
#min_year = day_year * hour_day * min_hour
#min_day = hour_day * min_hour
#
#tot_min = range(min_year / step)
#tot_min = np.array(tot_min)
##tot_day = np.arange()
#print tot_min[-1]
#
#tot_minDay = min_day / step
#print tot_minDay

#sun_vector = np.array([zeros()])

print "Begin calculation!!!"

min_frac = 60 / step

hour_year = 0

r = open("hourlyResult.dat", "w")
s = open("stepResult.dat", "w")
m = open("mirrorPosition.dat", "w")
gamma = 90
r.write("Hour_year Day Hour    G_on   G_o    decl   w_sunset   w_sunrise    w Light_hours  Theta_zeta   Gamma_s   Theta_t   Theta_l   Theta_i  f_end \n")
s.write("Day Hour Minute   G_on   G_o    decl   w_sunset   w_sunrise    w Light_hours  Theta_zeta   Gamma_s   Theta_t   Theta_l   Theta_i   f_end \n")
for day in range(1,days+1):
    G_on = G_sc * (1+0.033*cos(deg2rad(360*day/365)))
    B = (day-1) * 360/365
    E = 292.2 * (0.000075 + 0.001868 * cos(deg2rad(B)) - 0.032077 * sin(deg2rad(B)) - 0.014615 * cos(deg2rad(2*B)) - 0.04089 * sin(deg2rad(2*B)))
    decl = 23.45 * sin(deg2rad (360*(284+day)/365))
    cos_omega_sunset = -tan(deg2rad(latitude)) * tan(deg2rad(decl)) 
    omega_sunset = rad2deg(arccos(cos_omega_sunset))
    N=omega_sunset*2/15
    omega_sunrise = -omega_sunset
    H_o = 24*3600*G_sc/np.pi*G_on*(cos(deg2rad(latitude))*cos(deg2rad(decl))*cos(deg2rad(omega_sunset))+np.pi*omega_sunset/180*sin(deg2rad(latitude))*sin(deg2rad(decl)))
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
            n_end = 1-tan(deg2rad(theta_i))*H_abs/l_col
            tan_beta = tan(deg2rad(theta_zeta))*np.abs(cos(deg2rad(gamma*np.sign(gamma_s)-gamma_s)))
            beta_p = rad2deg(arctan(tan_beta))
            G_o = G_on*cos_theta_zeta
            
            if stepRes:

                s.write("%2d" % day) 
                s.write("%6d" % hour)
                s.write("%6.1f" % frac_min)
                s.write("%10.1f" % G_on)
                s.write("%10.1f" % G_o)
                s.write("%10.1f" % decl)
                s.write("%10.1f" % omega_sunset)
                s.write("%10.1f" % omega_sunrise)
                s.write("%10.2f" % w)
                s.write("%10.1f" % N)
                s.write("%10.1f" % theta_zeta)
                s.write("%10.1f" % gamma_s)
                s.write("%10.1f" % theta_t)
                s.write("%10.1f" % theta_l)
                s.write("%10.1f" % theta_i)
                s.write("%10.3f" % n_end)                
                s.write("\n")
                
                if theta_zeta < 0 or theta_zeta > 90:
                    
                    m.write("%2d" % day) 
                    m.write("%6d" % hour)
                    m.write("%6.1f" % frac_min)
                    m.write("\n")
                    
                    continue
                
                else:
                
                    m.write("%2d" % day) 
                    m.write("%6d" % hour)
                    m.write("%6.1f" % frac_min)
                    
                    m.write("%10.1f" % gamma_s)
                    m.write("%10.1f" % beta_p)
                    m.write("%10.1f" % theta)
                    m.write("%10.1f" % theta_zeta)
                    m.write("%10.1f" % theta_l)
                    m.write("%10.1f" % theta_i)
                    m.write("%10.1f" % theta_t)
                
                
                
                    for xCent in c_mirror:
                        
                        beta = inclMirror(theta_t, H_abs, xCent, 0)
                        c = coordMirror(W, xCent, l_col, beta, 0)
                        
                        m.write("  |   ")
                        m.write("%6.1f" % xCent)
                        m.write("%6.1f" % beta)
                        
 
                     #   np.savetxt(m, c, delimiter=",")
                    m.write("\n")
            
        if hourlyRes:
            r.write("%2d" % hour_year)

            r.write("%6d" % day)             
            r.write("%6d" % hour)
            r.write("%10.1f" % G_on)
            r.write("%10.1f" % G_o)
            r.write("%10.1f" % decl)
            r.write("%10.1f" % omega_sunset)
            r.write("%10.1f" % omega_sunrise)
            r.write("%10.1f" % w)
            r.write("%10.1f" % N)
            r.write("%10.1f" % theta_zeta)
            r.write("%10.1f" % gamma_s)
            r.write("%10.1f" % theta_t)
            r.write("%10.1f" % theta_l)
            r.write("%10.1f" % theta_i)
            r.write("%10.1f" % n_end)
            r.write("\n")
        
    

r.close()
s.close()   
m.close()



#Solar_time = 4*(L_st - longitude) + E + time_st