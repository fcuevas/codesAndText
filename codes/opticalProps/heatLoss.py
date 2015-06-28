# -*- coding: utf-8 -*-
"""
Created on Fri May 22 18:04:22 2015

@author: fcuevas
"""

import matplotlib.pylab as plt
import numpy as np

from optparse import OptionParser

parser = OptionParser()
parser.add_option('-a', '--nomAbs', action='store', dest='nomAbs', default=0.92,help='[-] Absorption at nominal incidence angle',type='float')
parser.add_option('-p', '--plot', action='store', dest='plotAbs', default=1,help='[-] Boolean value to plot the graph of the absorptance 0/NO 1/YES',type='int')
parser.add_option('-f', '--nameFileAbs', action='store', dest='nameFileAbs', default="absorption",help='[-] name of the absorptance file',type='string')
parser.add_option('-r', '--nameFileRefl', action='store', dest='nameFileRefl', default="reflection",help='[-] name of the reflectance file',type='string')
options, args = parser.parse_args()

nomAbs = options.nomAbs
plot = options.plotAbs
absFile = options.nameFileAbs
reflFile = options.nameFileRefl
    
def heatLoss(temp, u0, u1):
    hl = u0*temp + u1*temp**2
       
    return hl

#####################################################################################################    
#u0_r1 = -0.0163778913558
#u1_r1 = 0.000908004069539
#
#u0_r2 = -0.0224409527139
#u1_r2 = 0.00109407191883
#
#
#u0_r3 = -0.0288850470761
#u1_r3 = 0.00127942591542
#
#
#u0_r4 = -0.0356027202344
#u1_r4 = 0.00146365912627
#
#
#u0_r5 = -0.042496903598
#u1_r5 = 0.00164639197059


u0_r1 = 0.139228579525
u1_r1 = 0.0060791018135


u0_r2 = 0.161044231976
u1_r2 = 0.00613322065815


u0_r3 = 0.179900235755
u1_r3 = 0.00618238988244


u0_r4 = 0.196919584717
u1_r4 = 0.00622562940468


u0_r5 = 0.213018782546
u1_r5 = 0.00626196389554


hl1 = [] 
hl2 = [] 
hl3 = [] 
hl4 = [] 
hl5 = [] 
temp =[]
for t in range(100,250,1):
    a1 = heatLoss(t, u0_r1, u1_r1)
    a2 = heatLoss(t, u0_r2, u1_r2)
    a3 = heatLoss(t, u0_r3, u1_r3)
    a4 = heatLoss(t, u0_r4, u1_r4)
    a5 = heatLoss(t, u0_r5, u1_r5)
    
    temp.append(t)
    hl1.append(a1)
    hl2.append(a2)
    hl3.append(a3)
    hl4.append(a4)
    hl5.append(a5)

    
temperature = np.array(temp).reshape(len(temp),1)
hl1 = np.array(hl1).reshape(len(hl1),1)
hl2 = np.array(hl2).reshape(len(hl2),1)
hl3 = np.array(hl3).reshape(len(hl3),1)
hl4 = np.array(hl4).reshape(len(hl4),1)
hl5 = np.array(hl5).reshape(len(hl5),1)


if plot:    
  plt.xlabel("Temperature [$^\circ$C]", fontsize=18)
  plt.xticks(size=15)
  plt.ylabel("Heat loss [W/m]", fontsize=18)
  plt.yticks(size=15)
#  plt.text(2.0, 0.75, 'Nominal \nabsorptance: 0.96', fontsize=18)
  plt.plot(temp, hl1, linewidth=2, color="r")
  plt.plot(temp, hl2, linewidth=2, color="b")
  plt.plot(temp, hl3, linewidth=2, color="g")
  plt.plot(temp, hl4, linewidth=2, color="y")
  plt.plot(temp, hl5, linewidth=2, color="m")
  legend = plt.legend(("r:0.025 m","r:0.03 m","r:0.035 m","r:0.04 m","r:0.045 m"),loc="up left")
  plt.grid(True)
  plt.gcf().subplots_adjust(bottom=0.2)
  ltext = plt.gca().get_legend().get_texts()
  plt.setp(ltext,fontsize=16)
#  plt.ylim([0,1])
  #plt.show()
  plt.savefig("figures/heatLoss.jpg")
  
  