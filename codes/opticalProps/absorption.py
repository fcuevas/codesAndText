# -*- coding: utf-8 -*-
"""
Created on Fri May 22 18:04:22 2015

@author: fcuevas
"""

import matplotlib.pylab as plt
import numpy as np

from optparse import OptionParser

parser = OptionParser()
parser.add_option('-a', '--nomAbs', action='store', dest='nomAbs', default=0.96,help='[-] Absorption at nominal incidence angle',type='float')
parser.add_option('-p', '--plot', action='store', dest='plotAbs', default=1,help='[-] Boolean value to plot the graph of the absorptance 0/NO 1/YES',type='int')
parser.add_option('-f', '--nameFileAbs', action='store', dest='nameFileAbs', default="absorption",help='[-] name of the absorptance file',type='string')
parser.add_option('-r', '--nameFileRefl', action='store', dest='nameFileRefl', default="reflection",help='[-] name of the reflectance file',type='string')
options, args = parser.parse_args()

nomAbs = options.nomAbs
plot = options.plotAbs
absFile = options.nameFileAbs
reflFile = options.nameFileRefl
    
def absorptance(nomAbs,ang):
    angFact = 1 - 1.5879*10**(-3)*ang + 2.7314 * 10 **(-4)*ang**2 - 2.3026*10**(-5)*ang**3 + \
    9.0244*10**(-7)*ang**4 - 1.8*10**(-8)*ang**5 + 1.7734*10**(-10)*ang**6 - 6.9937*10**(-13)*ang**7
    absAng = nomAbs*angFact    
    return absAng

#####################################################################################################    
absFileT = absFile + ".dat"    
reflFileT = reflFile + ".dat"
out1 = open(absFileT,"w")
out2 = open(reflFileT,"w")
dim = 1
interp = "linear"
fak = 1.0
fakx = 0.17453
ad = 0.0
header1 = "{0:<6s} {1:<2s} {2:<10s} {3:<5} {4:<10s} {5:<10} {6:<10s} {7:<10} {8:<10s} {9:<10} {10:<10s} {11:<10}".format("name: ", absFile, "\ndimension: ", dim, "\ninterp: ", interp, "\nfak: ", fak, "\nfakx: ", fakx, "\nad: ", ad)
header2 = "{0:<6s} {1:<2s} {2:<10s} {3:<5} {4:<10s} {5:<10} {6:<10s} {7:<10} {8:<10s} {9:<10} {10:<10s} {11:<10}".format("name: ", reflFile, "\ndimension: ", dim, "\ninterp: ", interp, "\nfak: ", fak, "\nfakx: ", fakx, "\nad: ", ad)

out1.write(header1)    
out1.write(" \n \n ")

out2.write(header2)    
out2.write(" \n \n ")
##################################################################################################### 
    
absAng = [] 
angle =[]
for ang in range(0,90,1):
    a = absorptance(nomAbs,ang)
    
    angle.append(ang)
    absAng.append(a)

angle.append(90)
absAng.append(0)
    
angle = np.array(angle).reshape(len(angle),1)
absAng = np.array(absAng).reshape(len(absAng),1)

reflAng = 1- absAng

res1 = np.hstack((angle,absAng))
np.savetxt(out1,res1, fmt= ["%12.4e", "%12.4e"])

res2 = np.hstack((angle,reflAng))
np.savetxt(out2,res2, fmt= ["%12.4e", "%12.4e"])

if plot:    
  plt.xlabel("Angle [$^\circ$]", fontsize=18)
  plt.xticks(size=15)
  plt.ylabel("Absorptance [-]", fontsize=18)
  plt.yticks(size=15)
  plt.text(2.0, 0.7, 'Nominal \nabsorptance: 0.96', fontsize=18)
  plt.plot(angle, absAng, linewidth=3, color="r")
  legend = plt.legend(("Absorptance",),loc="center left")
  plt.grid(True)
  plt.gcf().subplots_adjust(bottom=0.2)
  ltext = plt.gca().get_legend().get_texts()
  plt.setp(ltext,fontsize=16)
  #plt.show()
  plt.savefig("figures/absorptance.jpg")
  plt.ylim([0,1])
  