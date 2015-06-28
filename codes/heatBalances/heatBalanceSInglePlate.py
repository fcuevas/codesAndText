#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from optparse import OptionParser
import matplotlib.pyplot as plt

# @todo Add parser option to variables

#----------------------------------------------------------------------
#Option parser
#----------------------------------------------------------------------
parser = OptionParser()
parser.add_option('-q', '--qSol', action='store', dest='qSol', default=1000.0,help='solar flux absorbed',type='float')
parser.add_option('-a', '--emiss_ext', action='store', dest='emiss_ext', default=0.9,help='emittance outer side of the plate',type='float')
parser.add_option('-m', '--h_convExt', action='store', dest='h_convExt', default=10.0,help='external heat transfer coefficient',type='float')
options, args = parser.parse_args()

qSol=options.qSol
emiss_ext=options.emiss_ext
h_convExt=options.h_convExt
#----------------------------------------------------------------------


############################ PLATE GEOMETRY ##################################
# x length
l1 = 1.0

# y length
l2 = 1.0

########################### AMBIENT CONDITIONS ################################

#Initial plate temperature T(K)
Tinic = 30 + 273

Tend = 250 + 273

#Ambient temperature T(K)
Tamb = 30 + 273

# sky temperature T(K)
Tsky = Tamb -11


############################ CONSTANTS #######################################

# constant
sigma = 5.67*10**(-8)

#############################################################################

# Area exposed to radiation and convection
A = l1*l2


#############################################################################
T_int = Tamb

HL_conv = []
HL_rad = []
surfTemp = []

for Tsurf in range(Tinic, Tend, 1):
    hl_rad = A*emiss_ext*sigma*(Tsurf**4 - Tsky**4)
    hl_conv = A*h_convExt * (Tsurf - Tamb)
    hl_tot = hl_conv + hl_rad
    
    HL_conv.append(hl_conv)
    HL_rad.append(hl_rad)
    surfTemp.append(Tsurf)
    
    if hl_tot > qSol:
        break
    
HL_conv = np.array(HL_conv)
HL_rad = np.array(HL_rad)
surfTemp = np.array(surfTemp)

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
plt.savefig("figures/balancePlate.jpg")
plt.xlim([25,90])
    