#!/usr/bin/env python
# -*- coding: utf-8 -*-
from numpy import dtype, loadtxt, savetxt, array, genfromtxt, where, all
import numpy as np
from scipy import *
from scipy import optimize
import matplotlib.pyplot as plt
import os
import StringIO

from optparse import OptionParser


from math import pi
from math import atan


## @package wind_test
# This script shows how to run different sets of parameters and compare the
# results in one plot.
#
# At first, pleaes note the definition of folder references. All file paths
# are always relative to the folder from which 'python' has been called.
# While this file itself is placed in a subfolder named python, it is called
# from the 'script' one folder above with the command: \n
# 'python ./python/pyexample.py' \n
# Thus, all path are read relative to the position of said 'script', and e.g.
# 'receiver' is called by './receiver' and not '../receiver'





 # ----------------------------------------------------------------------
# Option parser - 
# ----------------------------------------------------------------------
parser = OptionParser()
parser.add_option('-q', '--height', action='store', dest='col_height', help='Collector Height',type='float', default=7.4)
parser.add_option('-r', '--length', action='store', dest='col_length', help='Collector Length',type='float', default=100.0)
options, args = parser.parse_args()

variable = options.col_height
# ---------------------------------------------------------------------- 


me = "HL_nonEvTube" # 'me' is part of the file name for the results


# select the columns that you want to filter
# the reference column is used for interpolation and should be repeated in the
# selected columns if it is to appear in the output file
reference_column = "T3"

                 
selected_columns = array([
                 "node",
                 "T1",
                 "T3",
                 "T10",
                 "q_HL_int"])
             
columnsGrad = array([
                 "node",
                 "T1",
                 "T3",
                 "q_HL_int"])

# Geometrical definitions ##############################################

# absorber outer radio [m]
r3 = 0.035

# absorber inner radio
r2 = 0.033

# absorber tube thickness [m]
thickTube = 0.002


# glass envelope outer radio [m]
r5 = 0.06

# glass envelope inner radio [m]
r4 = 0.057

# mode evacuated 
mode_evac = 0

# End of geometrical definitions ##############################################


# Environmetal definitions ###################################################

# Ambient temperature [°C]
T_amb = 30

# Wind velocity [m/s]
v_wind = 2.5

I_b = 750

# End of environmetal definitions ############################################


# Fluid flow conditions ######################################################

# Mode calculation (1 given length, 2 given temperatures)
mode_calc = 1

# Initial temperature [°C]
T_init = 270

# Final temperature
l_tube1 = 1000
l_tube = l_tube1 - 1
N_node_abs = l_tube

# fluid identifier 
ident_htf = 2001

# mass flow [kg/s]
mdot_htf = 8

# mode pressure drop calculation (1 on, 0 off)
mode_dp = 0


# End fluid flow conditions ######################################################



# Physical properties of materials ###############################################


# Absorber emittance (quadratic equation, _0 + _1 * Tabs + _2 * Tabs²)
#epsilon_abs_0 = 6.2e-2
#epsilon_abs_1 = 0.0
#epsilon_abs_2 = 2.0e-7

epsilon_abs_0 = 6.2e-2
epsilon_abs_1 = 0.0
epsilon_abs_2 = 2.0e-7


# End of physical properties of materials ###############################################



# End of intermediate calculations


# String conversion

r2Str = str(r2)
r3Str = str(r3)
r4Str = str(r4)
r5Str = str(r5)
mode_evacStr = str(mode_evac)


T_ambStr = str(T_amb)
v_windStr = str(v_wind)
T_initStr = str(T_init)
l_tubeStr = str(l_tube)
N_node_absStr = str(N_node_abs)
I_bStr = str(I_b)


epsilon_abs_0Str = str(epsilon_abs_0)
epsilon_abs_1Str = str(epsilon_abs_1)
epsilon_abs_2Str = str(epsilon_abs_2)


mode_calcStr = str(mode_calc)
mode_dpStr = str(mode_dp)

ident_htfStr = str(ident_htf)
mdot_htfStr = str(mdot_htf)


# End of string conversion




os.system("./receiver_fcuevas"    +      " -r2 " + r2Str + \
                                         " -r3 " + r3Str + \
                                         " -r4 " + r4Str + \
                                         " -r5 " + r5Str + \
                                         " -mode_evac " + mode_evacStr + \
                                         
                                         " -T10 " + T_ambStr + \
                                         " -v_wind " + v_windStr + \
                                         " -I_b " + I_bStr + \
                                         
                                         " -mode_calc " + mode_calcStr + \
                                         " -mode_dp " + mode_dpStr + \
                                         " -T_init " + T_initStr + \
                                         " -l_tube " + l_tubeStr + \
                                         " -N_node_abs " + N_node_absStr + \
                                         " -mdot_htf " + mdot_htfStr + \
                                         " -ident_htf " + ident_htfStr + \
                                        
                                         " -epsilon_abs_0 " + epsilon_abs_0Str + \
                                         " -epsilon_abs_1 " + epsilon_abs_1Str + \
                                         " -epsilon_abs_2 " + epsilon_abs_2Str )
                                         
                                                                           

# --------------------------------------
# copy the complete result file
result_file = open('results/iteration_tube.txt')


results = genfromtxt(result_file,names=True,usecols=selected_columns)


copy_file   = open('resultsTRM/'+me+'_rabs_.txt', 'w')

res_file = open('resultsTRM/'+me+'.txt', 'w')

# print the header for the new file first
savetxt(copy_file,np.column_stack(selected_columns),fmt='%12s')
# and then the rest of the data
savetxt(copy_file,results,fmt='%12.6f')


print "python: selected columns copied to",copy_file.name, "!"		


totLength = results.size


column = 0
totRow=columnsGrad.size
#grad = zeros(totRow*(totLength-1))
#grad.shape = (totLength-1,totRow)

grad = zeros((totRow)*(totLength-1))
grad.shape = (totLength-1,totRow)

for columnName in columnsGrad:  

  
  i=0
  if (columnName == 'node'):
    while ( i < totLength-1):
      grad[i][0]=i
      i = i+1
    column = column + 1
  else:
    while ( i < totLength-1):

      grad[i][column] = results[columnName][i+1] - results[columnName][i]    
      i = i +1
    
    column = column + 1





copy_file.close()
result_file.close()



fit_curve = open('resultsTRM/'+me+'_rabs_.txt', 'r')

fit = genfromtxt(fit_curve,names=True)

column1 = 'T3'
column2 = 'T10'
column3 = 'q_HL_int'


Tabs = fit[column1]
Tamb = fit[column2]
x_axis = Tabs - Tamb
y_axis = fit [column3]


tempInit = Tabs[0]
tempEnd = Tabs[-1]
tempAmb = Tamb[0]

fitfunc = lambda p, x: p[0]*x + p[1]*x**2 
 # Distance to the target function:
errfunc = lambda p, x, y: fitfunc(p, x) - y
 # Initial guess for the parameters
#p0 = [1, 2, 3, 3, 4]
p0 = [1, 2] 
 # Fit the curve:
p1, success = optimize.leastsq(errfunc, p0[:], args=(x_axis, y_axis))

fitEval1 = fitfunc(p1,x_axis)

tot_HL_fit=0
tot_HL=0
tot=0
for count in range(0,totLength):
  temp=sqrt((y_axis[count]-fitEval1[count])**2)
  HL=y_axis[count]
  HL_fit=fitEval1[count]
  #print "values", y_axis[count], fitEval[count], temp
  tot=tot+temp
  tot_HL=tot_HL+HL
  tot_HL_fit=tot_HL_fit+HL_fit



sqDiff=tot/totLength
sqDiffTot=tot

DT_in = tempInit - tempAmb
DT_out = tempEnd - tempAmb



HL_corrModel2DT = (p1[0]*(DT_in + DT_out)/2 + p1[1]*((DT_in**2 + DT_in*DT_out +DT_out**2)/3))*totLength
HL_corrModel2 = (p1[0]*(tempInit + tempEnd)/2 + p1[1]*((tempInit**2 + tempInit*tempEnd +tempEnd**2)/3))*totLength

HL_corrModel3DT = (p1[0]*DT_out + p1[1]*DT_out**2)*totLength
HL_corrModel3 = (p1[0]*tempEnd + p1[1]*tempEnd**2)*totLength

diffHL_Model2 = (HL_corrModel2DT - tot_HL )/tot_HL*100
diffHL_Model3 = (HL_corrModel3DT - tot_HL )/tot_HL*100

coeff_file   = open('HL_results.dat', 'w')

coeff_file.write("Total length (m): " + str(totLength) + "\n")
coeff_file.write("\nTotal heat loss calculated with TRM \n HL (kW): " + str(tot_HL/1000) + "\n")

coeff_file.write("\nWall absorber temperatures\n")
coeff_file.write("\nT initial (°C): " + str(tempInit) + "\n")
coeff_file.write("T end (°C): " + str(tempEnd) + "\n")

coeff_file.write("\n*************************************************************************************************\n")

coeff_file.write("\nCurves fitting:\n")

coeff_file.write("\n*************************************************************************************************\n")
coeff_file.write("Fitting, curve (u0*DT + u1*DT^2), DT: Tabs - Tamb\n")
coeff_file.write("\n*************************************************************************************************\n")

coeff_file.write("\n")
coeff_file.write("Correlation coefficients\n")
#coeff_file.write(" -q_HL_0 "+str(p1[0])+" -q_HL_1 "+str(p1[1])+" -q_HL_2 "+str(p1[2])+" -q_HL_3 "+str(p1[3])+" -q_HL_4 "+str(p1[4]))
coeff_file.write("\n-u0_2: "+str(p1[0])+" \n-u1_2: "+str(p1[1]) + "\n")
coeff_file.write(" \nSquares difference (W/m): " + str(sqDiff))
coeff_file.write("\nSum squares differences (kW): " + str(sqDiffTot/1000) + "\n")

coeff_file.write("\nHeat loss ColSim correlations\n")
coeff_file.write("\nHeat loss model 2 (DT)\n HL_2 (kW): " + str(HL_corrModel2DT/1000) + "\n")
coeff_file.write("\nDifference HL TRM\n DiffHL_2 (%): " + str(diffHL_Model2) + "\n")

coeff_file.write("\nHeat loss model 3 (DT)\n HL_3 (kW): " + str(HL_corrModel3DT/1000) + "\n")
coeff_file.write("\nDifference HL TRM\n DiffHL_3 (%): " + str(diffHL_Model3) + "\n")

fitfunc = lambda p, x: p[0]*x + p[1]*x**4 
 # Distance to the target function:
errfunc = lambda p, x, y: fitfunc(p, x) - y

 # Initial guess for the parameters
p0 = [1, 2, 3, 4]

 # Fit the curve:
p1, success = optimize.leastsq(errfunc, p0[:], args=(x_axis, y_axis))

fitEval2 = fitfunc(p1,x_axis)

tot=0
for count in range(0,totLength):
  temp=sqrt((y_axis[count]-fitEval2[count])**2)
  #print "values", y_axis[count], fitEval[count], temp
  tot=tot+temp

sqDiff=tot/totLength
sqDiffTot=tot

HL_corrModel1DT = (p1[0]*(DT_in + DT_out)/2 + (DT_out**4 + DT_out**3*DT_in + DT_out**2*DT_in**2 + DT_out*DT_in**3 + DT_in**4)/5*p1[1])*totLength
HL_corrModel1 =( p1[0]*(tempInit + tempEnd)/2 + (tempEnd**4 + tempEnd**3*tempInit + tempEnd**2*tempInit**2 + tempEnd*tempInit**3 + tempInit**4)/5*p1[1])*totLength

HL_corrModel4DT = (p1[0]*DT_out + p1[1]*DT_out**4)*totLength
HL_corrModel4 = (p1[0]*tempEnd + p1[1]*tempEnd**4)*totLength

diffHL_Model1 = (HL_corrModel1DT - tot_HL )/tot_HL*100
diffHL_Model4 = (HL_corrModel4DT - tot_HL )/tot_HL*100


coeff_file.write("\n*************************************************************************************************\n")
coeff_file.write("\nFitting, curve (u0*DT + u1*DT^4), DT: Tabs - Tamb\n")
coeff_file.write("\n*************************************************************************************************\n")
  
coeff_file.write("Correlation coefficients\n")
#coeff_file.write(" -q_HL_0 "+str(p1[0])+" -q_HL_1 "+str(p1[1])+" -q_HL_2 "+str(p1[2])+" -q_HL_3 "+str(p1[3])+" -q_HL_4 "+str(p1[4]))
coeff_file.write("\n-u0_4: "+str(p1[0])+" \n-u1_4: "+str(p1[1]) + "\n")
coeff_file.write(" \nSquares difference (W/m): " + str(sqDiff))
coeff_file.write("\nSum squares differences (kW): " + str(sqDiffTot/1000) + "\n")

coeff_file.write("\nHeat loss ColSim correlations\n")
coeff_file.write("\nHeat loss model 1 (DT)\n HL_1 (kW): " + str(HL_corrModel1DT/1000) + "\n")
coeff_file.write("\nDifference HL TRM\n DiffHL_1 (%): " + str(diffHL_Model1) + "\n")

coeff_file.write("\nHeat loss model 4 (DT)\n HL_4 (kW): " + str(HL_corrModel4DT/1000) + "\n")
coeff_file.write("\nDifference HL TRM\n DiffHL_4 (%): " + str(diffHL_Model4) + "\n")
# end of while-loop  
      
############################################################################################


