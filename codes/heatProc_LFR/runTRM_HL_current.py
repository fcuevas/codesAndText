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
                 
                 
u0_2 =0.422312437131
u1_2 =0.00321680777512

u0_4 =1.22406732289
u1_4 =7.21499121751e-09

N_div = 1

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
T_init =486.374

# Final temperature
l_tube1 =50.000000

l_tube = l_tube1-1

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



copy_file.close()
result_file.close()



fit_curve = open('resultsTRM/'+me+'_rabs_.txt', 'r')

fit = genfromtxt(fit_curve,names=True)

column1 = 'T3'
column2 = 'T10'
column3 = 'q_HL_int'
column4 = 'T1'


Tabs = fit[column1]
Tamb = fit[column2]
x_axis = Tabs - Tamb
y_axis = fit[column3]
Tfluid = fit[column4]


tempInit = Tabs[0]
tempEnd = Tabs[-1]
tempAmb = Tamb[0]

tempEndFluid = Tfluid[-1]

tot_HL=0
for count in range(0,totLength):
  HL=y_axis[count]
  tot_HL=tot_HL+HL



DT_in = tempInit - tempAmb
DT_out = tempEnd - tempAmb



HL_corrModel2DT = (u0_2*(DT_in + DT_out)/2 + u1_2*((DT_in**2 + DT_in*DT_out +DT_out**2)/3))*totLength
HL_corrModel2 = (u0_2*(tempInit + tempEnd)/2 + u1_2*((tempInit**2 + tempInit*tempEnd +tempEnd**2)/3))*totLength

HL_corrModel3DT = (u0_2*DT_out + u1_2*DT_out**2)*totLength
HL_corrModel3 = (u0_2*tempEnd + u1_2*tempEnd**2)*totLength

diffHL_Model2 = (HL_corrModel2DT - tot_HL )/tot_HL*100
diffHL_Model3 = (HL_corrModel3DT - tot_HL )/tot_HL*100


coeff_file   = open('HL_results.dat', 'w')

coeff_file.write("Total length (m): " + str(totLength) + "\n")
coeff_file.write("\nTotal heat loss calculated with TRM \n HL (kW): " + str(tot_HL/1000) + "\n")

coeff_file.write("\nWall absorber temperatures\n")
coeff_file.write("\nT initial (°C): " + str(tempInit) + "\n")
coeff_file.write("T end (°C): " + str(tempEnd) + "\n")
coeff_file.write("Tf end (°C): " + str(tempEndFluid) + "\n")

coeff_file.write("\n*************************************************************************************************\n")

coeff_file.write("\nCurves fitting:\n")

coeff_file.write("\n*************************************************************************************************\n")
coeff_file.write("Fitting, curve (u0*DT + u1*DT^2), DT: Tabs - Tamb\n")
coeff_file.write("\n*************************************************************************************************\n")


coeff_file.write("\nHeat loss ColSim correlations\n")
coeff_file.write("\nHeat loss model 2 (DT)\n HL_2 (kW): " + str(HL_corrModel2DT/1000) + "\n")
coeff_file.write("\nDifference HL TRM\n DiffHL_2 (%): " + str(diffHL_Model2) + "\n")

coeff_file.write("\nHeat loss model 3 (DT)\n HL_3 (kW): " + str(HL_corrModel3DT/1000) + "\n")
coeff_file.write("\nDifference HL TRM\n DiffHL_3 (%): " + str(diffHL_Model3) + "\n")


tot_HL=0
for count in range(0,totLength):
  HL=y_axis[count]
  tot_HL=tot_HL+HL


HL_corrModel1DT = (u0_4*(DT_in + DT_out)/2 + (DT_out**4 + DT_out**3*DT_in + DT_out**2*DT_in**2 + DT_out*DT_in**3 + DT_in**4)/5*u1_4)*totLength
HL_corrModel1 =( u0_4*(tempInit + tempEnd)/2 + (tempEnd**4 + tempEnd**3*tempInit + tempEnd**2*tempInit**2 + tempEnd*tempInit**3 + tempInit**4)/5*u1_4)*totLength

HL_corrModel4DT = (u0_4*DT_out + u1_4*DT_out**4)*totLength
HL_corrModel4 = (u0_4*tempEnd + u1_4*tempEnd**4)*totLength

diffHL_Model1 = (HL_corrModel1DT - tot_HL )/tot_HL*100
diffHL_Model4 = (HL_corrModel4DT - tot_HL )/tot_HL*100


coeff_file.write("\n*************************************************************************************************\n")
coeff_file.write("\nFitting, curve (u0*DT + u1*DT^4), DT: Tabs - Tamb\n")
coeff_file.write("\n*************************************************************************************************\n")


coeff_file.write("\nHeat loss ColSim correlations\n")
coeff_file.write("\nHeat loss model 1 (DT)\n HL_1 (kW): " + str(HL_corrModel1DT/1000) + "\n")
coeff_file.write("\nDifference HL TRM\n DiffHL_1 (%): " + str(diffHL_Model1) + "\n")

coeff_file.write("\nHeat loss model 4 (DT)\n HL_4 (kW): " + str(HL_corrModel4DT/1000) + "\n")
coeff_file.write("\nDifference HL TRM\n DiffHL_4 (%): " + str(diffHL_Model4) + "\n")
# end of while-loop  
      
############################################################################################


