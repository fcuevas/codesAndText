# -*- coding: utf-8 -*-
"""
Created on Sat Jan  3 17:18:40 2015

@author: fcuevas
"""
from numpy import cos, sin, tan, arccos, arctan, deg2rad, rad2deg, genfromtxt, arcsin
import numpy as np


def sunPos(lat,day,hour,minute):
    decl = 23.45 * sin(deg2rad (360*(284+day)/365))
    w=-(12-(hour+minute/60.))*15
    cos_theta_zeta = cos(deg2rad(lat))*cos(deg2rad(decl))*cos(deg2rad(w))+sin(deg2rad(lat))*sin(deg2rad(decl))
    theta_zeta = rad2deg(arccos(cos_theta_zeta))

            
    gamma_s = np.sign(w)*np.abs(rad2deg(arccos((cos(deg2rad(theta_zeta))*sin(deg2rad(lat))-sin(deg2rad(decl)))/(sin(deg2rad(theta_zeta))*cos(deg2rad(lat))))))

            
    return theta_zeta, gamma_s

#########################################################################################
    
def colAngles(theta_zeta, gamma_s):
    
    theta_t=rad2deg(arctan(sin(deg2rad(gamma_s))*tan(deg2rad(theta_zeta))))
    theta_l=rad2deg(arctan(cos(deg2rad(gamma_s))*tan(deg2rad(theta_zeta))))
    theta_i=rad2deg(arcsin(cos(deg2rad(gamma_s))*sin(deg2rad(theta_zeta))))
    
    return theta_t, theta_l, theta_i
    

#########################################################################################

def getEta_opt(theta_t, theta_i, IAMfile):
    values = genfromtxt(IAMfile, comments="#")
    eta_opt0 = values[0,1]
    
    for n in range(len(values)):
        if (theta_t - values[n,0])<= 0:
          eta_opt_t= values[n,1] + (values[n-1,1] - values[n,1])*(theta_t - values[n-1,0])/(values[n,0] - values[n-1,0])
          #print values[n,1], values[n-1,1], values[n,0], values[n-1,0]
          break
    
    for n in range(len(values)):      
        if (theta_i - values[n,0])<= 0:
          eta_opt_l= values[n,2] + (values[n-1,2] - values[n,2])*(theta_i - values[n-1,0])/(values[n,0] - values[n-1,0])
          #print values[n,2], values[n-1,2], values[n,0], values[n-1,0]
          break
          
    etaOpt = (eta_opt_t * eta_opt_l)/eta_opt0
    return etaOpt #, eta_opt0, eta_opt_t, eta_opt_l

#########################################################################################

def getDNI(pos, solRadFile):
    result_file = open(solRadFile)
    results = genfromtxt(result_file, skip_header=2)
    DNI = results[pos,4]
    
    return DNI

#########################################################################################

fileDebug = False


if fileDebug == True:

    result_file = open('crucero4.txt')
    latitude = -22.27
    
    results = genfromtxt(result_file, skip_header=2)
    
    totValues = len(results)
    totValuesDay=totValues/365
    totValuesHour=totValues/8760
    totValuesHourDay=totValuesHour/24
    
    # Delete first two columns
    resultsFilt=results[:,2:]
    
    #Step of the weather data (minutes)
    step = 10
    
    day = np.array(np.zeros(totValues)).reshape(totValues,1)
    hourYear = np.array(np.zeros(totValues)).reshape(totValues,1)
    hourDay = np.array(np.zeros(totValues)).reshape(totValues,1)
    minDay = np.array(np.zeros(totValues)).reshape(totValues,1)
    dayCount = 0
    dayNum = 1
    
    for d in xrange(totValues):
      
      if dayCount < totValuesDay:
        day[d] = dayNum
        dayCount = dayCount +1
      else:
        dayNum = dayNum+1
        day[d] = dayNum
        dayCount = 1
        
      
    hourCount = 0
    hourNum = 1
    hourNumDay=0
    minCount=0
    
    for d in xrange(totValues):
      
      if hourCount < totValuesHour:
        hourYear[d] = hourNum
        hourCount = hourCount +1
        hourDay[d]=hourNumDay   
        
        minCount = minCount + step
        minDay[d] = minCount
       
      else:
        hourNum=hourNum+1
        hourYear[d] = hourNum
        minCount=0
        minDay[d] = minCount
        
        if hourNumDay < 23:
          hourNumDay=hourNumDay+1
          
        else:
          hourNumDay=0
        hourDay[d]=hourNumDay 
        hourCount = 1
    
    t1=np.hstack((day,hourYear))
    t1=np.hstack((t1,hourDay))
    t1=np.hstack((t1,minDay))
    
    
    weatherFile = np.hstack((t1,resultsFilt))
    
    
    
    #########################################################################################
    days=365
    
    min_frac = 60 / step
    
    hour_year = 0
    
    # Solar Constant, radiation outside the atmosphere (W/m2)
    G_sc = 1367 
    
    #Heigth of the absorber center (m)
    H_col=7
    
    #Lenght of the collector (m)
    l_col=100
    
    W_col=8.0
    
    G_onL = []
    B_L = []
    E_L = []
    decl_L = []
    
    gamma = 90
    
    totValues = 365*24*6
    print totValues
    
    theta_z = np.array(np.zeros(totValues)).reshape(totValues,1)
    theta_t = np.array(np.zeros(totValues)).reshape(totValues,1)
    theta_l = np.array(np.zeros(totValues)).reshape(totValues,1)
    theta_i = np.array(np.zeros(totValues)).reshape(totValues,1)
    gamma_s = np.array(np.zeros(totValues)).reshape(totValues,1)
    eta    = np.array(np.zeros(totValues)).reshape(totValues,1)
    DNI    = np.array(np.zeros(totValues)).reshape(totValues,1)
    countTot=0
    
    for day in range(1,days+1):
        
        for hour in range(0,24):
            hour_year = hour_year+1
            for minute in range(0,min_frac):
                
                frac_min = step*minute
                theta_zeta, gamma_sT = sunPos(latitude, day, hour, frac_min)
    
                theta_tT, theta_lT, theta_iT = colAngles(theta_zeta, gamma_sT)
                
                theta_z[countTot]=theta_zeta
                theta_t[countTot]=theta_tT
                theta_l[countTot]=theta_lT
                theta_i[countTot]=theta_iT
                gamma_s[countTot]=gamma_sT
                DNI = getDNI(countTot, "crucero4.txt")
                
                print theta_zeta, gamma_sT, theta_tT, theta_iT, DNI, countTot
                if theta_zeta < 0 or theta_zeta > 90 or np.isnan(gamma_sT):
                    eta[countTot] = 0.0
                    
                else:
                    eta[countTot] = getEta_opt(theta_tT, theta_iT, "IAM199.dat")
                
                countTot=countTot+1
                
                
    solarAngles=np.hstack((theta_z,gamma_s))   
    anglesLFC = np.hstack((theta_t,theta_l))  
    anglesLFC = np.hstack((anglesLFC,theta_i)) 
    
    #
    print eta[0:300]
    #etaOptAnnual = sum(eta*weatherFile[:,5]*step/60*W_col)
    #print etaOptAnnual
    #etaOpt_Annual = sum(eta*W_col*weatherFile[:,5])
    
        
        
    #########################################################################################
    
    daylyIntegration = False
    
    if daylyIntegration:
    
      dniDay=[]
      dhiDay=[]
      ghiDay=[]
      dniTemp=0
      ghiTemp=0
      dhiTemp=0
      t=1
    
      step=10
      stepDay=1440/step
      stepDayPrint=stepDay  
      
      for d in xrange(len(weatherFile)):
          ghiTemp=ghiTemp + weatherFile[d,4]
          dniTemp=dniTemp + weatherFile[d,5]
          dhiTemp=dhiTemp + weatherFile[d,6]
          
          if d > stepDayPrint:
      
        	ghiDay.append(ghiTemp)
        	dniDay.append(dniTemp)
        	dhiDay.append(dhiTemp)
        	ghiTemp=0
        	dniTemp=0
        	dhiTemp=0
        	t=t+1 
        	stepDayPrint= stepDay*t
    	
          if d == (len(weatherFile)-1):
        	ghiDay.append(ghiTemp)
        	dniDay.append(dniTemp)
        	dhiDay.append(dhiTemp)
        	
          
          
    #########################################################################################
    
    hourlyIntegration = False
    
    if hourlyIntegration:
    
      dniHour=[]
      dhiHour=[]
      ghiHour=[]
      dniTemp=0
      ghiTemp=0
      dhiTemp=0
      t=1
    
      step=10
      stepHour=60/step
      stepHourPrint=stepHour
    
      for d in xrange(len(weatherFile)):
          ghiTemp=ghiTemp + weatherFile[d,4]
          dniTemp=dniTemp + weatherFile[d,5]
          dhiTemp=dhiTemp + weatherFile[d,6]
          
          if d > stepHourPrint:
      
        	ghiHour.append(ghiTemp)
        	dniHour.append(dniTemp)
        	dhiHour.append(dhiTemp)
        	ghiTemp=0
        	dniTemp=0
        	dhiTemp=0
        	t=t+1 
        	stepHourPrint= stepHour*t
    	
          if d == (len(weatherFile)-1):
        	ghiHour.append(ghiTemp)
        	dniHour.append(dniTemp)
        	dhiHour.append(dhiTemp)
    	
    #########################################################################################	
    
    
