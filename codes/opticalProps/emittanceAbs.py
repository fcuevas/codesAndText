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
    
def emittance(temp, abs_0, abs_1, abs_2):
    emTemp = abs_0 + abs_1*temp + abs_2*temp**2
       
    return emTemp

#####################################################################################################    
epsilon_abs_0 = 6.2e-2
epsilon_abs_1 = 0.0
epsilon_abs_2 = 2.0e-7

#epsilon_abs_0 = 0.0767
#epsilon_abs_1 = 0.0006
#epsilon_abs_2 = 1.0e-6
    
absEm = [] 
temp =[]
for t in range(0,250,1):
    a = emittance(t, epsilon_abs_0, epsilon_abs_1, epsilon_abs_2)
    
    temp.append(t)
    absEm.append(a)

    
temperature = np.array(temp).reshape(len(temp),1)
absEm = np.array(absEm).reshape(len(absEm),1)


if plot:    
  plt.xlabel("Temperature [$^\circ$C]", fontsize=18)
  plt.xticks(size=15)
  plt.ylabel("Emittance [-]", fontsize=18)
  plt.yticks(size=15)
#  plt.text(2.0, 0.75, 'Nominal \nabsorptance: 0.96', fontsize=18)
  plt.plot(temp, absEm, linewidth=3, color="r")
  legend = plt.legend(("Emittance",),loc="center left")
  plt.grid(True)
  plt.gcf().subplots_adjust(bottom=0.2)
  ltext = plt.gca().get_legend().get_texts()
  plt.setp(ltext,fontsize=16)
  plt.ylim([0,1])
  #plt.show()
  plt.savefig("figures/emittance.jpg")
  
  