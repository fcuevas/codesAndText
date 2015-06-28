#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
import numpy as np

from optparse import OptionParser

parser = OptionParser()
parser.add_option('-B', '--Width', action='store', dest='Width', default=0.5,help='[m] Width of the primary field mirrors',type='float')
parser.add_option('-N', '--Nmirr', action='store', dest='Nmirr', default=10,help='Number of mirrors',type='int')
parser.add_option('-H', '--heigth', action='store', dest='heigth', default=4.5,help='[m] Heigth of the absorber tube',type='float')
parser.add_option('-D', '--distX', action='store', dest='distX', default=0.15,help='[m] Distance between mirrors, transversal plane',type='float')
parser.add_option('-L', '--length', action='store', dest='length', default=3.0,help='[m] Length of the mirrors',type='float')
parser.add_option('-d', '--distY', action='store', dest='distY', default=0.1,help='[m] Distance between mirrors, longitudinal plane',type='float')
parser.add_option('-f', '--NmirrY', action='store', dest='NmirrY', default=3,help='Number of mirrors in y direction',type='int')
parser.add_option('-r', '--radii', action='store', dest='radii', default=0.35,help='[m] Radii of absorber tube',type='float')
parser.add_option('-a', '--ApSec', action='store', dest='ApSec', default=0.2,help='[m] Aperture of the secondary mirror',type='float')
parser.add_option('-s', '--hCol', action='store', dest='hCol', default=0.5,help='[m] Distance of the primary field about the level of the ground',type='float')
parser.add_option('-i', '--insThick', action='store', dest='insThick', default=0.05,help='[m] Thickness of the insulation',type='float')
parser.add_option('-t', '--casThick', action='store', dest='casThick', default=0.001,help='[m] Thickness of the casing',type='float')
parser.add_option('-M', '--optMirr', action='store', dest='optMirr', default=1,help='Mirror option (1 / silvered glass, 2 / aluminum)',type='int')
parser.add_option('-c', '--optAbs', action='store', dest='optAbs', default=2,help='Absorber option (1 / tube, Solkote coating, 2 / Evacuated tube)',type='int')
#parser.add_option('-k', '--numModul', action='store', dest='numModul', default=1,help='Number of modules',type='int')
parser.add_option('-z', '--totAp', action='store', dest='totAp', default=10000,help='Total aperture',type='int')
parser.add_option('-k', '--costWorkSteal', action='store', dest='costWorkSteal', default=5.0,help='[US$/m2] Cost of working steal',type='float')

options, args = parser.parse_args()

def squareProfile(case):
  if case == 1:
    prof_w = 0.05
    prof_h = 0.05
    prof_t = 0.002  
    
  elif case == 2:
    prof_w = 0.05
    prof_h = 0.05
    prof_t = 0.003

  elif case == 3:
    prof_w = 0.01
    prof_h = 0.01
    prof_t = 0.002   
    
  elif case == 4:
    prof_w = 0.03
    prof_h = 0.03
    prof_t = 0.002  
    
  elif case == 5:
    prof_w = 0.04
    prof_h = 0.04
    prof_t = 0.002  
    
  elif case == 6:
    prof_w = 0.06
    prof_h = 0.06
    prof_t = 0.002  
    
  else:
    print "Case not defined!!!"
    Aprof = np.nan  
    
  return prof_w, prof_h, prof_t


def areaProfile(case):
   
  prof_w, prof_h, prof_t = squareProfile(case)  
  Aprof = prof_w * prof_h - ( (prof_w - 2*prof_t) * (prof_h - 2*prof_t))
  return Aprof

def contourProfile(case):
  prof_w, prof_h, prof_t = squareProfile(case) 
  cProf = 4*prof_w
  return cProf
  
def tubeProfile(case):
  if case == 1:
    d_ext = 0.0381
    e = 0.0015
    
  if case == 2:
    d_ext = 0.04445
    e = 0.0015
    
  if case == 3:
    d_ext = 0.0508
    e = 0.002
    
  if case == 4:
    d_ext = 0.0635
    e = 0.002
    
  if case == 5:
    d_ext = 0.0762
    e = 0.002
    
  if case == 6:
    d_ext = 0.0889
    e = 0.002
 
  return d_ext, e
 
def areaTube(case):
  d_ext, e = tubeProfile(case)
  aTube = np.pi*d_ext
  
  return aTube
  
 
debug = True
#debug = False

B = options.Width
N = options.Nmirr
f = options.NmirrY
H = options.heigth
D = options.distX
L = options.length
d = options.distY
r = options.radii
h = options.hCol
apSec = options.ApSec
insThick = options.insThick
casThick = options.casThick
optMirr = options.optMirr
optAbs = options.optAbs
totAp = options.totAp
costWork = options.costWorkSteal
#totMod = options.numModul

USd = 580


#
Amirr = N*B*L*f
totMod = totAp / Amirr

# 
colW = N*B + (N-2)*D

#
modL = L*f + d*(f-1)
colL = modL * totMod
#
mirrH = h

colH_angle = 60
colH = np.sin(np.deg2rad(colH_angle))*H

# steel density (kg/m3)
rho_steel = 7650
rho_aluminum = 2800
###########################################
############## STRUCTURE ####################

profPlate_w = 0.52
profPlate_h = 0.007
profPlate_t = 0.0007
AprofPlate = profPlate_w * profPlate_t

###########################################

AprofBar = areaProfile(1)
CprofBar= contourProfile(1)
lengthBarS = modL
volBarS = modL*AprofBar
weigthBarS = volBarS * rho_steel
totWeigthBarS = 2*weigthBarS 
totContBarS = 2 * CprofBar * modL

if debug:
  print "## Values side bar ######"
  print "Length barS (m): ", lengthBarS 
  print "Volume barS (m): ", volBarS 
  print "Weigth barS (m): ", weigthBarS 
  print "Total weigth (m): ", totWeigthBarS 
  print "#########################\n"

###########################################

###########################################

lengthBarF = colW
volBarF = colW*AprofBar
weigthBarF = volBarF * rho_steel
totWeigthBarF = f*weigthBarF 
totContBarF = f * CprofBar * modL

if debug:
  print "## Values frontal bar ######"
  print "Length barF (m): ", lengthBarF 
  print "Volume barF (m): ", volBarF 
  print "Weigth barF (m): ", weigthBarF 
  print "Total weigth (m): ", totWeigthBarF 
  print "#########################\n"
###########################################

###########################################

lengthBarH = colH
volBarH = colH*AprofBar
weigthBarH = volBarH * rho_steel
totWeigthBarH = 2*weigthBarH 
totContBarH = 2 * CprofBar * modL

if debug:
  print "## Values lateral bar (support receiver) ######"
  print "Length barH (m): ", lengthBarH 
  print "Volume barH (m): ", volBarH 
  print "Weigth barH (m): ", weigthBarH 
  print "Total weigth (m): ", totWeigthBarH 
  print "#########################\n"
###########################################

###########################################
AprofMirr = areaProfile(1)
CprofMirr= contourProfile(1)
lengthBarM = modL
volBarM = modL * AprofMirr
weigthBarM = volBarM * rho_steel
totWeigthBarM = N * weigthBarM * f 
totContBarM = N * CprofMirr * modL * f

if debug:
  print "## Values mirror support ######"
  print "Length barM (m): ", lengthBarM 
  print "Volume barM (m): ", volBarM 
  print "Weigth barM (m): ", weigthBarM 
  print "Total weigth (m): ", totWeigthBarM 
  print "#########################\n"
###########################################

###########################################
AprofLeg = areaProfile(1)
CprofLeg= contourProfile(1)
lengthLeg = mirrH
volLeg = mirrH*AprofLeg
weigthLeg = volLeg * rho_steel
totWeigthLeg = 3 * f * weigthLeg
totContBarLeg = 3 * f * CprofLeg 

if debug:
  print "## Values leg bar ######"
  print "Length barL (m): ", lengthLeg 
  print "Volume barL (m): ", volLeg 
  print "Weigth barL (m): ", weigthLeg 
  print "Total weigth (m): ", totWeigthLeg
  print "#########################\n"
###########################################


###########################################
AprofFix = areaProfile(3)/2
CprofFix = contourProfile(3)
lengthMirrFix = modL
volProfFix = modL*AprofFix
weigthFix = volProfFix * rho_steel
totWeigthFix = N*2*weigthFix
totContBarFix = N * 2 * CprofFix

if debug:
  print "## Values mirror fixation ######"
  print "Length mirror fixation (m): ", lengthMirrFix
  print "Volume mirror fixation (m): ", volProfFix 
  print "Weigth mirror fixation (m): ", weigthFix 
  print "Total weigth (m): ", totWeigthFix 
  print "#########################\n"

###########################################
#############CASING ######################

rCas_int = apSec/2+insThick
rCas_ext = rCas_int + casThick
areaCas = 2*np.pi*(rCas_ext - rCas_int)/2
volCas = areaCas * modL
weigthCas = volCas * rho_aluminum

if debug:
  print "## Values casing ######"
  print "Radii casing (m): ", rCas_int
  print "Area casing (m): ", areaCas
  print "Volume casing (m): ", volCas
  print "Weigth casing (m): ", weigthCas
  
  
rIns_int = apSec/2
rIns_ext = apSec/2+insThick    
areaIns = 2*np.pi*(rIns_ext - rIns_int)
volIns = areaIns * modL

totAreaStr = 
totWeigthStr = totWeigthLeg + totWeigthBarM + totWeigthBarH + totWeigthBarF + totWeigthBarS  
costStrMod = totWeigthStr * costWork
costStrCol = costStrMod * totMod
###########################################
############## MIRRORS ####################

if optMirr == 1:
  print "Mirror option: Silvered glass"
  
  # (US$/m2)  
  costGlass = 6.75
  costRefl = 15
  costMirr = costRefl + costGlass
  #costMirr = Amirr * costMirr  
  
   
elif optMirr == 2:
  print "Mirror option: Aluminum, Mirage"
  
  # US$/m2
  costMirr = 10  
  #costMirr = Amirr * costMirr
  
elif optMirr == 3:
  print "Mirror option: Aluminum, Miro-Sun"
  
  # US$/m2
  costMirr = 21.5    
  
  
costMirrMod = Amirr * costMirr
costMirrCol = costMirrMod * totMod
###########################################
############## ABSORBER ####################

if optAbs == 1:
  print "Absorber option: stainless steel, Solkote coating"
  
  costTube = 4.5
  aTube = 2*np.pi*r
  # USD/gal
  costGalPaint = 88
  # m2/gal
  totArea = 46
  # Estimated, efficiency of the process of painting (how much paint can be effectively used)
  effPaint = 0.6
  costPaint = (costGalPaint * aTube / totArea) / effPaint
  costAbs = costTube + costPaint
  
  
  

if optAbs == 2:
  print "Absorber option: evacuated tube"  
  costAbs = 160

costAbsMod = costAbs * modL
costAbsCol = costAbsMod * totMod
###########################################
############## SECONDARY MIRROR ####################

# Almeco- S090
costAlu = 30000/USd
areaSec = apSec*np.pi*modL
costSec = areaSec * costAlu

costSecMod = costSec * modL
costSecCol = costSecMod * totMod
###########################################
############## GLASS PLATE ####################

costGlass = 6.75  
areaGlass = apSec * modL
costGlass = areaGlass * costGlass

costGlassMod = costGlass * modL
costGlassCol = costGlassMod * totMod
########################################################################

totCostMod = costGlassMod + costSecMod + costAbsMod + costMirrMod + costStrMod
totCostCol = costGlassCol + costSecCol + costAbsCol + costMirrCol + costStrCol






########################################################################
print "\n#################Resume collector#######################"
print "Total aperture (m2): ", totAp
print "Total modules (-): ", totMod
print "Aperture module (m2): ",Amirr 

####################### RESUME COSTS ###################################

print "\n######## Structure ######"
print "Weigth module (kg): ", totWeigthStr
print "Weigth collector (kg): ", totWeigthStr * totMod
print "Cost structure module (US$): ", costStrMod
print "Cost structure collector (US$): ", costStrCol

print "\n######## Mirrors ######"
print "Cost mirrors / module (US$): ", costMirrMod
print "Cost mirrors collector (US$): ", costMirrCol

print "\n######## Absorber ######"
print "Cost absorber / module (US$): ", costAbsMod
print "Cost absorber collector (US$): ", costAbsCol

print "\n######## Secondary ######"
print "Cost secondary / module (US$): ", costSecMod
print "Cost secondary collector (US$): ", costSecCol

print "\n######## Glass plate ######"
print "Cost glass plate / module (US$): ", costGlassMod
print "Cost glass plate collector (US$): ", costGlassCol

print "\n######## Total cost ######"
print "Total cost module (US$): ", totCostMod
print "Total cost collector (US$): ", totCostCol
print "Total cost /m2 (US$/m2): ", totCostCol/totAp
