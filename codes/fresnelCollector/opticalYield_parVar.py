# -*- coding: utf-8 -*-
"""
Created on Sat Jan  3 17:18:40 2015

@author: fcuevas
"""
from numpy import genfromtxt, cos, sin, tan, arccos, arcsin, arctan, sqrt, deg2rad, rad2deg
import numpy as np
from scipy.interpolate import interp1d
from sunPos import sunPos, declination, G_onSun, w_sunset, omega
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata

def H_o(lat, day):
    decl = declination(day)
    G_on = G_onSun(day)
    omega_sunset = w_sunset(lat, day)
    H_o = 24*3600/np.pi*G_on*(cos(deg2rad(lat))*cos(deg2rad(decl))*cos(deg2rad(omega_sunset))+np.pi*omega_sunset/180*sin(deg2rad(lat))*sin(deg2rad(decl)))
    return H_o
    
    
def I_o(lat,day,hour,minute, step):
    decl = declination(day)
    G_on = G_onSun(day)
    I_o = 12*3600/np.pi*G_on*(cos(deg2rad(lat))*cos(deg2rad(decl))*(sin(deg2rad(omega(hour,minute+step))) - sin(deg2rad(omega(hour,minute))))+np.pi*(omega(hour,minute+step)-omega(hour,minute))/180*sin(deg2rad(lat))*sin(deg2rad(decl)))
    return I_o
    
    
    
def colAngles(theta_zeta, gamma_s):
    
    theta_l=rad2deg(arctan(cos(deg2rad(gamma_s))*tan(deg2rad(theta_zeta))))
    theta_i=rad2deg(arcsin(cos(deg2rad(gamma_s))*sin(deg2rad(theta_zeta))))

    angInt = [90,180,270,360]   
    if int(theta_zeta) in angInt:
        theta_t = 90
        
    else:
        theta_t=rad2deg(arctan(np.abs(sin(deg2rad(gamma_s)))*tan(deg2rad(theta_zeta))))
        
    return theta_t, theta_l, theta_i    
#########################################################################################
    
def getEta_opt(theta_t, theta_i, IAMfile):
    IAM = genfromtxt(IAMfile, comments="#")
    eta_opt0 = IAM[0,1]
    
#    print IAM[0,1], theta_t, theta_i
    
    interp_t = interp1d(IAM[:,0], IAM[:,1])
    interp_l = interp1d(IAM[:,0], IAM[:,2])
    
    etaOpt_t = np.abs(interp_t(theta_t)) * np.cos(np.deg2rad(theta_t))
    etaOpt_l = np.abs(interp_l(theta_i)) * np.cos(np.deg2rad(theta_i))
    
    etaOpt = (etaOpt_t * etaOpt_l)/eta_opt0
    return etaOpt

def getIAM_values(IAMfile):
    IAM = genfromtxt(IAMfile, comments="#")
    eta_opt0 = IAM[0,1]
    IAM[:,1] = IAM[:,1]*np.cos(np.deg2rad(IAM[:,0]))
    IAM[:,2] = IAM[:,2]*np.cos(np.deg2rad(IAM[:,0]))
    return eta_opt0, IAM[:,0], IAM[:,1], IAM[:,2]
    
def inclMirror(theta_t, h_abs, x_m, Dz):
    beta = (theta_t - rad2deg(arctan(x_m/(h_abs + Dz))))/2
    return beta


def coordMirror(W, xCent, xLen, beta, foc):

    x1 = (xCent - W/2.)*cos(deg2rad(beta))
    y1 = -colLength/2.
    z1 = (xCent - W/2.)*sin(deg2rad(beta))

    x2 = (xCent + W/2.)*cos(deg2rad(beta))
    y2 = -colLength/2.
    z2 = (xCent + W/2.)*sin(deg2rad(beta))
    c1 = np.array([[x1,y1,z1],[x2,y2,z2], [x2,-y2,z2], [x1,-y1,z1]])
    
    return c1
    
    
def heatLoss(temp, u0, u1):
    hl = u0*temp + u1*temp**2
       
    return hl
#########################################################################################

collectorResults = True

#case = "highTec"
case = "lowTec"
temp = np.array([100, 150, 200, 250])

if case == "highTec":
    IAM_file = "IAM_highTec.dat"
    dirFigs = "figuresHighTec/"
    u0_r3 = -0.0288850470761
    u1_r3 = 0.00127942591542
    
    
    
else:
    IAM_file = "IAM_lowTec.dat"
    dirFigs = "figuresLowTec/"
    u0_r3 = 0.179900235755
    u1_r3 = 0.00618238988244


heatL = heatLoss(temp, u0_r3, u1_r3)

result_file = open('crucero4.txt')
latitude = -22.3

if collectorResults:
    Ap = 6.0
    colLength = 100
    
    H_abs = 4.5
    N = 10
    W = 0.6
    d = 0.15
    
    x_mirror = 0
    
    if N%2 == 0.0:
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
    
#########################################################################################
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
t2=np.hstack((t1,hourDay))
t3=np.hstack((t2,minDay))


weatherFile = np.hstack((t3,resultsFilt))

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
  
  if collectorResults:
      
      hourlyRadRes = open("hourlyCollectorResults.dat", "w")
      headerHR = "{0:^12} {1:^12} {2:^12} {3:^12} {4:^12} {5:^12}\
    {6:^12} {7:^12} {8:^12} {9:^12} {10:^12} {11:^12} {12:^12} {13:^12} {14:^12} {15:^12} {16:^12}"\
    .format("Hour_year", "hour_day", "DNI", "DHI", "GHI", "G_on",\
    "Kt", "theta_z", "gamma_s", "theta_zT", "theta_zL",  "theta_zI", "eta", "qDisp", "qDispTot", "qDispTotApEff", "ApEff    \n")
    
  else:
      
      hourlyRadRes = open("hourlyRadiationResults.dat", "w")
      headerHR = "{0:^12} {1:^12} {2:^12} {3:^12} {4:^12} {5:^12}\
    {6:^12} {7:^12} "\
    .format("Hour_year", "DNI", "DHI", "GHI", "G_on",\
    "Kt", "theta_z", "gamma_s   \n")
      
    
  hourlyRadRes.write(headerHR)

  hourDay= np.array(np.zeros(totValues)).reshape(totValues,1)
  dayYear= np.array(np.zeros(totValues)).reshape(totValues,1)
  dniHourly= np.array(np.empty(totValues)).reshape(totValues,1) 
  dhiHourly= np.array(np.empty_like(dniHourly))
  ghiHourly= np.array(np.empty_like(dniHourly))
  globHourly= np.array(np.empty_like(dniHourly))
  Hour= np.array(np.empty_like(dniHourly))
  angThetaz= np.array(np.empty_like(dniHourly))
  angGamma= np.array(np.empty_like(dniHourly))

  if collectorResults:
      
      angThetaT= np.array(np.empty_like(dniHourly))
      angThetaL= np.array(np.empty_like(dniHourly))
      angThetaI= np.array(np.empty_like(dniHourly))
      eta= np.array(np.empty_like(dniHourly))
      beta= np.array(np.empty(totValues*N)).reshape(totValues, N)
      cosMirr = np.array(np.empty_like(beta))

  dniTemp=0
  ghiTemp=0
  dhiTemp=0
  t=1

  step=10
  stepHour=60/step
  stepHourPrint=stepHour
  countHour = 0
  
  

  for d in xrange(len(weatherFile)):
      ghiTemp=ghiTemp + weatherFile[d,4] * step/60
      dniTemp=dniTemp + weatherFile[d,5] * step/60
      dhiTemp=dhiTemp + weatherFile[d,6] * step/60
      
      day = 1 + countHour / 24
      hour = countHour % 24
      
      
      if d > stepHourPrint:
          dayYear[countHour] = day
          hourDay[countHour] = hour
          ghiHourly[countHour] = ghiTemp
          dniHourly[countHour] = dniTemp
          dhiHourly[countHour] = dhiTemp
          ghiTemp=0
          dniTemp=0
          dhiTemp=0
          t=t+1 
          stepHourPrint= stepHour*t
          Hour[countHour] = countHour
          globHourly[countHour] = I_o(latitude, day, hour, 0, 60)
          angThetaz[countHour], angGamma[countHour] = sunPos(latitude, day, hour)
          
          if collectorResults:
          
              angThetaT[countHour] , angThetaL[countHour], angThetaI[countHour]=colAngles(angThetaz[countHour], angGamma[countHour])
                        
              if angThetaz[countHour] < 0 or angThetaz[countHour] > 90:
                  eta[countHour] = 0.0
                  beta[countHour,:] = 0.0
                  
              else:
                  
                  mirrID = 0
                  for xCent in c_mirror:
                      
                      beta[countHour,mirrID] = inclMirror(angThetaT[countHour], H_abs, xCent, 0)
                      mirrID=mirrID+1
                  
                  if angThetaI[countHour] < 0.0:
                      angThetaI[countHour] = 0.0
                      eta[countHour] = getEta_opt(angThetaT[countHour], angThetaI[countHour], IAM_file)
                      
                  else:
                      eta[countHour] = getEta_opt(angThetaT[countHour], angThetaI[countHour], IAM_file)
                  
                        
          countHour = countHour + 1
       
	
      if d == (len(weatherFile)-1):
          ghiHourly[countHour] = ghiTemp
          dniHourly[countHour] = dniTemp
          dhiHourly[countHour] = dhiTemp
          Hour[countHour] = countHour
          hourDay[countHour] = hour 
          dayYear[countHour] = day
          countHour = countHour + 1
          

       
  globHourly = globHourly/3600    
  Kt = ghiHourly / globHourly 
  
  resTemp = np.hstack((Hour,dayYear))
  resTemp = np.hstack((resTemp,hourDay))
  resTemp = np.hstack((resTemp,ghiHourly))
  resTemp = np.hstack((resTemp, dniHourly))
  resTemp = np.hstack((resTemp, dhiHourly))
  resTemp = np.hstack((resTemp, globHourly))
  resTemp = np.hstack((resTemp, Kt))
   
  resTemp = np.hstack((resTemp, angThetaz))
  resTemp = np.hstack((resTemp, angGamma))
  
  totDNI = np.sum(dniHourly)
  totDHI = np.sum(dhiHourly)
  totGHI = np.sum(ghiHourly)
  totG_on = sum(globHourly[globHourly > 0])
  
  if collectorResults:
            
      l=0
      for ang in angThetaT:
          if beta[l,0] != 0.0:
              cosMirr[l] = cos(deg2rad(ang - beta[l])) 
              l = l+1
          else:
              cosMirr[l,:] = 0.0
              l=l+1
              
      ApEff = W * np.sum(cosMirr, axis=1)
      ApEff = np.array(ApEff).reshape(len(eta),1)
        
      contAnn = np.sum(cosMirr, axis=0)  
      qDispTotApEff = eta * dniHourly * ApEff * colLength      
      qDisp = eta*dniHourly 
      qDispTot = eta * dniHourly * Ap 
      
      
      qUsefTot = qDispTot - heatL
#      qUsefTot = eta * dniHourly * Ap - heatL
      print heatL
            
      resTemp = np.hstack((resTemp, angThetaT))
      resTemp = np.hstack((resTemp, angThetaL))
      resTemp = np.hstack((resTemp, angThetaI))
      resTemp = np.hstack((resTemp, eta))
      resTemp = np.hstack((resTemp, qDisp))
      resTemp = np.hstack((resTemp, qDispTot))
      resTemp = np.hstack((resTemp, qDispTotApEff))
    
      resTemp = np.hstack((resTemp, ApEff))
      resTemp = np.hstack((resTemp, beta))
      resTemp = np.hstack((resTemp, cosMirr))
      totqDisp = np.sum(qDisp)
      totqUsef = np.sum(qUsefTot[qUsefTot > 0])
      totqDispAp = np.sum(qDispTot)
      
      annOptEff = totqDisp/totDNI * 100
     
      np.savetxt(hourlyRadRes,resTemp, fmt=' '.join(['%8d','%8d','%8d','%12.1f','%12.1f','%12.1f','%12.1f','%12.3f','%12.2f','%12.2f','%12.2f','%12.2f','%12.2f','%12.2f','%12.1f','%12.1f','%12.1f','%12.1f'] + ["|"] + ['%12.2f']*N + ["|"] + ['%12.2f']*N))
      totRes = np.array([totDNI/10**3, totqDisp/10**3, annOptEff, totqDispAp/10**3, totqUsef/10**3])

  else:
      
      np.savetxt(hourlyRadRes,resTemp, fmt=['%8d','%12.1f','%12.1f','%12.1f','%12.1f','%12.3f','%12.2f','%12.2f'])
      totRes = np.array([totDNI/10**3, totDHI/10**3, totGHI/10**3, totG_on/10**3])
  
  print totRes
  
#  histDNI, binDNI = np.histogram(dniHourly,bins=np.arange(1,1202,100))
#  histThetaZ, binThetaZ = np.histogram(angThetaz, bins = np.arange(0,91,10))
#  histEtaOpt, binEtaOpt = np.histogram(eta, bins = np.arange(0.01,1.01,0.1))
#  histQavail, binQavail = np.histogram(qDisp, bins = np.arange(1,1002,100))
#  
#  
##########################################################################################	
#  
#  plt.xlabel("Radiation range [W/m2]")
#  plt.ylabel("Frequency [hours/year]")
#  plt.hist(dniHourly,binDNI, facecolor="green")
#  plt.grid(True)
#  plt.savefig("figures/histDNI_Chile.jpg")
#  plt.close()  
#
##########################################################################################	
#
#  plt.xlabel("Zenithal angle []")
#  plt.ylabel("Frequency [hours/year]")
#  plt.hist(angThetaz,binThetaZ, facecolor="green")
#  plt.grid(True)
#  plt.savefig("figures/histTheta_Chile.jpg")
#  plt.close() 
#
##########################################################################################  
#  name ="histEtaOpt_Chile.jpg" 
#  fileDir = dirFigs + name  
#  
#  plt.xlabel("Optical efficiency [-]")
#  plt.ylabel("Frequency [hours/year]")
#  plt.hist(eta,binEtaOpt, facecolor="green")
#  plt.grid(True)
#  plt.savefig(fileDir)
#  plt.close() 
#  
#  #########################################################################################	
#  name ="histHeat_Chile.jpg" 
#  fileDir = dirFigs + name
#  
#  plt.xlabel("Heat absorbed [W/m2]")
#  plt.ylabel("Frequency [hours/year]")
#  plt.hist(qDisp,binQavail, facecolor="green")
#  plt.grid(True)
#  plt.savefig(fileDir)
#  plt.close() 
#  
#  ###########################################################################
#
#etaOpt0, angles, IAM_t, IAM_l = getIAM_values(IAM_file)
#
#name ="IAM_collector.jpg" 
#fileDir = dirFigs + name
#
#plt.xlabel(" Zenith angle [$^\circ$]")
#plt.ylabel("Optical efficiency  [W/m]")
#plt.plot(angles, IAM_t, linestyle="--", linewidth=3, label="transversal")
#plt.plot(angles, IAM_l, linestyle=":", linewidth=4, label="longitudinal")
#legend = plt.legend(loc="center left")
#plt.grid(True)
#plt.savefig(fileDir)
#plt.xlim([0,90])
#
#plt.close()
#
#dniHM = dniHourly.reshape(365,24).T
#
#plt.pcolormesh(dniHM)
#plt.xlabel(" Day of the year")
#plt.ylabel("Hour of the day")
#plt.title("DNI (W/m2)")
#plt.axis([0, 365, 0,23])
#plt.colorbar()
#plt.savefig("figures/DNI_HM.jpg")
#plt.close()
#
#totDat = 24*6
#dniDet = weatherFile[:,5].reshape(365,totDat).T
#plt.pcolormesh(dniDet)
#plt.xlabel(" Day of the year")
#plt.ylabel("Hour of the day")
#plt.title("DNI (W/m2)")
#plt.axis([0, 365, 0,totDat])
#plt.colorbar()
#plt.savefig("figures/DNI_det.jpg")
#plt.close()
#
#
#name ="eta_HM.jpg" 
#fileDir = dirFigs + name
#
#etaHM = eta.reshape(365,24).T
#plt.pcolormesh(etaHM)
#plt.xlabel(" Day of the year")
#plt.ylabel("Hour of the day")
#plt.title("Collector efficiency (-)")
#plt.axis([0, 365, 0,23])
#plt.colorbar()
#plt.savefig(fileDir)
#plt.close()                  
#
#
#name ="thetaZ_HM.jpg" 
#fileDir = dirFigs + name
#
#theta_zHM = angThetaz.reshape(365,24).T
#plt.pcolormesh(theta_zHM)
#plt.xlabel(" Day of the year")
#plt.ylabel("Hour of the day")
#plt.title("Azimuthal angle ($^\circ$)")
#plt.axis([0, 365, 0,23])
#plt.colorbar()
#plt.savefig(fileDir)
#plt.close()  
#
#
#name ="qDisp_HM.jpg" 
#fileDir = dirFigs + name
#
#qDispHM = qDisp.reshape(365,24).T
#plt.pcolormesh(qDispHM)
#plt.xlabel(" Day of the year")
#plt.ylabel("Hour of the day")
#plt.title("Heat available (W/m2)")
#plt.axis([0, 365, 0,23])
#plt.colorbar()
#plt.savefig(fileDir)
#plt.close() 
#
#
#name ="qDispTot_HM.jpg" 
#fileDir = dirFigs + name
#
#qDispTotHM = qDispTot.reshape(365,24).T
#plt.pcolormesh(qDispTotHM)
#plt.xlabel(" Day of the year")
#plt.ylabel("Hour of the day")
#plt.title("Heat available collector aperture (W/m)")
#plt.axis([0, 365, 0,23])
#plt.colorbar()
#plt.savefig(fileDir)
#plt.close() 
#
#
#name ="qUsefTot_HM.jpg" 
#fileDir = dirFigs + name
#
#qUsefTotHM = qUsefTot.reshape(365,24).T
#plt.pcolormesh(qDispTotHM)
#plt.xlabel(" Day of the year")
#plt.ylabel("Hour of the day")
#plt.title("Useful heat (W/m)")
#plt.axis([0, 365, 0,23])
#plt.colorbar()
#plt.savefig(fileDir)
#plt.close() 
