#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from optparse import OptionParser
from scipy.optimize import fsolve 
import matplotlib.pyplot as plt

# @todo Add parser option to variables

#----------------------------------------------------------------------
#Option parser
#----------------------------------------------------------------------
parser = OptionParser()
parser.add_option('-w', '--width', action='store', dest='l3', default=0.0032,help='width of the plate',type='float')
parser.add_option('-k', '--cond', action='store', dest='k', default=1.1,help='conductivity of the plate',type='float')
parser.add_option('-q', '--qSol', action='store', dest='qSol', default=1000.0,help='solar flux absorbed',type='float')
parser.add_option('-a', '--emiss_ext', action='store', dest='emiss_ext', default=0.9,help='emittance outer side of the plate',type='float')
parser.add_option('-b', '--emiss_int', action='store', dest='emiss_int', default=0.9,help='emittance inner side of the plate',type='float')
parser.add_option('-c', '--emiss_inf', action='store', dest='emiss_inf', default=0.25,help='emittance bottom plate',type='float')
#parser.add_option('-s', '--T_inf', action='store', dest='T_inf', default=523.0,help='temperature of the bottom plate',type='float')
parser.add_option('-l', '--l_d', action='store', dest='l_d', default=0.2,help='distance between plates',type='float')
parser.add_option('-m', '--h_convExt', action='store', dest='h_convExt', default=10.0,help='external heat transfer coefficient',type='float')
options, args = parser.parse_args()

l3=options.l3
k=options.k
qSol=options.qSol
emiss_ext=options.emiss_ext
emiss_int=options.emiss_int
emiss_inf=options.emiss_inf
#T_inf=options.T_inf
l_d=options.l_d
h_convExt=options.h_convExt
#----------------------------------------------------------------------

################################ AIR PROPERTIES (K) ##########################################

# air density. T(K)
def rho(T):
  return 2.69489*pow(10,-6)*pow(T,2) - 4.40475*pow(10,-3)*T +2.23821

# air specific heat. T(K)  
def Cp(T):
  return 2.07758*pow(10,-4)*pow(T,2) - 3.98720*pow(10,-2)*T +9.97359*pow(10,2)
  
def ka(T):
  return -2.11915*pow(10,-8)*pow(T,2) + 8.63029*pow(10,-5)*T + 1.6924*pow(10,-3)
  
def mu(T):
  return -1.666668*pow(10.,-11.)*pow(T,2.) + 5.39357*pow(10.,-8.)*T + 3.94702*pow(10.,-6.)
  
#############################################################################################


# Function to calculate Nusselt number between parallel plates
def Nu_nat_ParallPlate(Ra):  
    if (Ra<10e5):
      a= 0.195*pow(Ra,0.25)
    
    else:
      a= 0.068*pow(Ra,1./3.)
    return a
    
# Function to calculte Grashof number    
def Gr_l(T1, T2, nu, L):
  g=9.81
  r= g*1/((T1+T2)/2)*(T1-T2)*pow(L,3)/pow(nu,2)
  return r

  
# Function to calculate convective heat transfer coefficient   
def h_conv(Nu,D, k):
  h = Nu*k/D
  return h  
    

############################ PLATES DIMENSION ################################## 


############################ PLATE GEOMETRY ##################################
# x length
l1 = 1.0

# y length
l2 = 1.0
  
########################### AMBIENT CONDITIONS ################################

#Initial plate temperature T(K)
Tinic = 30 + 273

#Ambient temperature T(K)
Tamb = 30 + 273

# sky temperature T(K)
Tsky = Tamb -11

Tend = 500 + 273

############################ CONSTANTS #######################################

# constant
sigma = 5.67*10**(-8)

#############################################################################

# Area exposed to radiation and convection
A = l1*l2

#############################################################################

incr = 0.01

qAbs = qSol*0.85
qAbs_cov = qSol*0.02


Tabs = []
HL_conv = []
HL_rad = []

for T_inf in range(Tamb+10,Tend,1):
    
    
    T_int = Tamb
    Tabs.append(T_inf)    
    res = 1000
    sig=1
    res_b = 1

    while abs(res) > 1e-2 and sig > 0:      
      
      T_int = T_int + incr      
      Tf = (T_int + T_inf)/2.
          
      if emiss_inf ==0.0 or emiss_int==0.0:
        qRad1 = 0.0
        
      else:
        
        qRad1 = A*sigma*(T_inf**4 - T_int**4)/(1/emiss_int + 1/emiss_inf - 1)
      
      if T_int > T_inf:
        qConv1 = A*ka(Tf)*(T_inf - T_int)/l_d
        
      else:
        
        nu = mu(Tf)/rho(Tf)
        Pr = mu(Tf)*Cp(Tf)/ka(Tf)
        
        Gr = Gr_l(T_inf,T_int,nu,l_d)
        Ra = Pr*Gr
        
        if ( Ra < 10e4):
          
          continue
    
        else:
          
          Nu=Nu_nat_ParallPlate(Ra)
          h = h_conv(Nu, l_d, ka(Tf))
          qConv1 = A*h*(T_inf-T_int)
        
      q_int = qConv1  + qRad1 

      def eqs(p):
        
        T_ext = p
        
        qConv2 = A*h_convExt*(T_ext - Tamb)
    
        qRad2 = A*sigma*emiss_ext*(T_ext**4 - Tsky**4)
      
        q_ext = qRad2 + qConv2 - q_int
        
        return (q_ext)
        
      T_ext = fsolve(eqs, (0))
      
      
      qConv2 = A*h_convExt*(T_ext - Tamb)    
      qRad2 = A*sigma*emiss_ext*(T_ext**4 - Tsky**4)    
      q_ext = qRad2 + qConv2 
       
      qCond = -k*A*(T_int - T_ext)/l3 
      
      res = q_int + qCond
      sig = res * res_b
      res_b = res
      
      

    HL_conv.append(qConv1)
    HL_rad.append(qRad1)

    
    if q_int > qSol:
        break
  
  
  
HL_conv = np.array(HL_conv)
HL_rad = np.array(HL_rad)
surfTemp = np.array(Tabs)

HL_tot = HL_conv + HL_rad

###########################################################################

plt.xlabel("Absorber temperature [$^\circ$C]", fontsize=18)
plt.xticks(size=15)
plt.ylabel("Heat loss  [W]", fontsize=18)
plt.yticks(size=15)
plt.plot(surfTemp-273, HL_conv, linestyle="--", linewidth=3)
plt.plot(surfTemp-273, HL_rad, linestyle=":", linewidth=4)
plt.plot(surfTemp-273, HL_tot, linewidth=1.5)
legend = plt.legend(("Convection", "Radiation", "Total heat loss"),loc="upper left")
plt.grid(True)
ltext = plt.gca().get_legend().get_texts()
plt.setp(ltext[:],fontsize=16)
#plt.annotate()
#plt.show()
plt.savefig("figures/balanceParallelPlates.jpg")
plt.xlim([25,surfTemp[-1]-273])
#############################################################################
