# -*- coding: utf-8 -*-
"""
Created on Sat Jan  3 17:18:40 2015

@author: fcuevas
"""

import numpy as np

from sunPos import sunPos


#########################################################################################


result_file = open('crucero4.txt')
latitude = -22.27


#########################################################################################
results = np.genfromtxt(result_file, skip_header=2)

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

daylyIntegration = False

if daylyIntegration:
  totValues = 365  


  dniDay= np.array(np.zeros(totValues)).reshape(totValues,1) 
  dhiDay= np.array(np.zeros(totValues)).reshape(totValues,1) 
  ghiDay= np.array(np.zeros(totValues)).reshape(totValues,1) 
  dniTemp=0
  ghiTemp=0
  dhiTemp=0
  t=1

  step=10
  stepDay=1440/step
  stepDayPrint=stepDay  
  countDay = 0
  
  for d in xrange(len(weatherFile)):
      ghiTemp=ghiTemp + weatherFile[d,4]
      dniTemp=dniTemp + weatherFile[d,5]
      dhiTemp=dhiTemp + weatherFile[d,6]
      
      if d > stepDayPrint:
          ghiDay[countDay] = ghiTemp
          dniDay[countDay] = dniTemp
          dhiDay[countDay] = dhiTemp
          ghiTemp=0
          dniTemp=0
          dhiTemp=0
          t=t+1 
          stepDayPrint= stepDay*t
          countDay = countDay +1
	
      if d == (len(weatherFile)-1):
          ghiDay[countDay] = ghiTemp
          dniDay[countDay] = dniTemp
          dhiDay[countDay] = dhiTemp
          countDay = countDay +1
	
      
      
#########################################################################################

hourlyIntegration = True

if hourlyIntegration:
  totValues = 8760 


  dniDay= np.array(np.zeros(totValues)).reshape(totValues,1) 
  dhiDay= np.array(np.zeros(totValues)).reshape(totValues,1) 
  ghiDay= np.array(np.zeros(totValues)).reshape(totValues,1) 
  dniTemp=0
  ghiTemp=0
  dhiTemp=0
  t=1

  step=10
  stepHour=60/step
  stepHourPrint=stepHour
  countHour = 0
  
  

  for d in xrange(len(weatherFile)):
      ghiTemp=ghiTemp + weatherFile[d,4]
      dniTemp=dniTemp + weatherFile[d,5]
      dhiTemp=dhiTemp + weatherFile[d,6]
      
      day = 1 + countHour / 24
      hour = countHour % 24
      
      theta_z, gamma_s = sunPos(latitude, day, hour)
      
      
      
      if d > stepHourPrint:
          ghiDay[countHour] = ghiTemp
          dniDay[countHour] = dniTemp
          dhiDay[countHour] = dhiTemp
          ghiTemp=0
          dniTemp=0
          dhiTemp=0
          t=t+1 
          stepHourPrint= stepHour*t
          countHour = countHour + 1
	
      if d == (len(weatherFile)-1):
          ghiDay[countHour] = ghiTemp
          dniDay[countHour] = dniTemp
          dhiDay[countHour] = dhiTemp
          countHour = countHour + 1
	
#########################################################################################	


