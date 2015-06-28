# -*- coding: utf-8 -*-
"""
Created on Fri May 22 18:04:22 2015

@author: fcuevas
"""

import numpy as np
import matplotlib.pylab as plt

from optparse import OptionParser

parser = OptionParser()

parser.add_option('-n', '--refrInd', action='store', dest='refrInd', default=1.52,help='[-] Refraction index of glass cover',type='float')
parser.add_option('-t', '--thickness', action='store', dest='thickness', default=0.0032,help='[m] thickness of glass cover',type='float')
parser.add_option('-k', '--extCoeff', action='store', dest='extCoeff', default=32,help='[-] extinction coefficient of glass cover',type='float')
parser.add_option('-f', '--nameFile', action='store', dest='nameFile', default="trans",help='[-] name of the transmittance file',type='string')
options, args = parser.parse_args()

n2 = options.refrInd
e = options.thickness
k = options.extCoeff
transFile = options.nameFile

def ang2(n1,n2,ang):
    theta1 = np.deg2rad(ang)
    theta2 = n1/n2*np.sin(theta1)    
    return theta1, theta2
    
def tauAbs(e,K,ang):
    tau = np.exp(-K*e/np.cos(ang))
    return tau
    
def reflectance(n1, n2, ang):
    theta1, theta2 = ang2(n1,n2,ang)
    refl_per = (np.sin(theta2 - theta1))**2 / (np.sin(theta2 + theta1))**2
    refl_par = (np.tan(theta2 - theta1))**2 / (np.tan(theta2 + theta1))**2
    refl = (refl_per + refl_par)/2
    return refl_per, refl_par, refl
    
    
    
#de reflPlate(tAbs)

def transPlate(tau,n1,n2,ang):

    refl_per, refl_par, refl = reflectance(n1,n2,ang) 
    
    T_per = tau * (1-refl_per)/(1+refl_per) * (1-refl_per**2)/(1-(refl_per*tau)**2)
    T_par = tau * (1-refl_par)/(1+refl_par) * (1-refl_par**2)/(1-(refl_par*tau)**2)
    T = 0.5 * (T_per + T_par)
    return T_per, T_par, T
    
def reflPlate(tau,n1,n2,ang):

    refl_per, refl_par, refl = reflectance(n1,n2,ang) 
    T_per, T_par, T = transPlate(tau,n1,n2,ang)
    
    R_per = refl_per*(1+tau*T_per)
    R_par = refl_par*(1+tau*T_par)
    R = 0.5 * (R_per + R_par)
    return R_per, R_par, R
    
def absPlate(tau,n1,n2,ang):

    refl_per, refl_par, refl = reflectance(n1,n2,ang) 
    
    A_per = (1-tau)*(1-refl_per)/(1-refl_per*tau)
    A_par = (1-tau)*(1-refl_par)/(1-refl_par*tau)
    A = 0.5 * (A_per + A_par)
    return A_per, A_par, A
 
n1 = 1.0


#####################################################################################################    
transFileT = transFile + ".dat"    
out = open(transFileT,"w")
dim = 1
interp = "linear"
fak = 1.0
fakx = 0.17453
ad = 0.0
header = "{0:<6s} {1:<2s} {2:<10s} {3:<5} {4:<10s} {5:<10} {6:<10s} {7:<10} {8:<10s} {9:<10} {10:<10s} {11:<10}".format("name: ", transFile, "\ndimension: ", dim, "\ninterp: ", interp, "\nfak: ", fak, "\nfakx: ", fakx, "\nad: ", ad)

out.write(header)    
out.write(" \n \n ")

##################################################################################################### 



plateTrans = [] 
plateAbs = [] 
plateRefl = []
angle = []

for ang in range(1,91,1):
    
    theta1, theta2 = ang2(n1,n2,ang)
    tAbs = tauAbs(e,k,theta2)
    
    Tpl_per, Tpl_par, Tpl = transPlate(tAbs, n1, n2, ang)
    Rpl_per, Rpl_par, Rpl = reflPlate(tAbs, n1, n2, ang)
    Apl_per, Apl_par, Apl = absPlate(tAbs, n1, n2, ang)
    
    ref_per, ref_par, r = reflectance(n1,n2,ang)
    
  
    angle.append(ang)
    plateTrans.append(Tpl)
    plateRefl.append(Rpl)
    plateAbs.append(Apl)
    

#angle.append(90)
#plateTrans.append(0)

angle = np.array(angle).reshape(len(angle),1)
plateTrans = np.array(plateTrans).reshape(len(plateTrans),1)

res = np.hstack((angle,plateTrans))
np.savetxt(out,res, fmt= ["%12.4e", "%12.4e"])    
    
    
plt.xlabel("Angle [$^\circ$]", fontsize=18)
plt.xticks(size=15)
plt.ylabel("[-]", fontsize=18)
plt.yticks(size=15)
plt.plot(angle, plateTrans, linestyle="--", linewidth=3)
plt.plot(angle, plateRefl, linestyle=":", linewidth=4)
plt.plot(angle, plateAbs, linewidth=1.5)
legend = plt.legend(("Transmitance", "Reflectance", "Absorptance"),loc="center left")
plt.grid(True)
plt.gcf().subplots_adjust(bottom=0.25)
ltext = plt.gca().get_legend().get_texts()
plt.setp(ltext[:],fontsize=16)
#plt.show()
plt.savefig("figures/balanceGlassPlate.jpg")
plt.xlim([0,90])
plt.ylim([0,1])