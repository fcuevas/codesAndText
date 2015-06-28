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

from sunPos import sunPos


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


me = "heatProc_LFC" # 'me' is part of the file name for the results


# select the columns that you want to filter
# the reference column is used for interpolation and should be repeated in the
# selected columns if it is to appear in the output file
reference_column = "T3"

                 
selected_columns = array([
                 "node",
                 "T3",
                 "T10",
                 "q_HL_int"])
             
columnsGrad = array([
                 "node",
                 "T3",
                 "q_HL_int"])

# Geometrical definitions ##############################################

# absorber outer radio [m]
r3 = 0.045

# absorber inner radio
r2 = 0.033

l_tube = 500

# absorber tube thickness [m]
thickTube = 0.002

r6 = 0.198
s6_7 = 0.005
s8_9 = 0.04
r9 = 0.2585

s_plate = 0.003
b_plate = 0.39
area_cor = 0.53

# glass envelope outer radio [m]
r5 = 0.06

# glass envelope inner radio [m]
r4 = 0.057

# mode evacuated 
mode_evac = 1

# End of geometrical definitions ##############################################


# Environmetal definitions ###################################################

# Ambient temperature [°C]
T_amb = 25

# Wind velocity [m/s]
v_wind = 2.5

# End of environmetal definitions ############################################


# Fluid flow conditions ######################################################

# Mode calculation (1 given length, 2 given temperatures)
mode_calc = 1



# Initial temperature [°C]
T_init = 100

# End temperature [°C]
#T_end = 150

# fluid identifier 
ident_htf = 3000
p_htfBar = 5
p_htf = 500000 

# mass flow [kg/s]
mdot_htf = 5

# mode pressure drop calculation (1 on, 0 off)
mode_dp = 1

q7g_SolAbs = 0 
q9_SolAbs = 0  

q6_SolAbs = 0


# End fluid flow conditions ######################################################



# Physical properties of materials ###############################################


# Absorber emittance (quadratic equation, _0 + _1 * Tabs + _2 * Tabs²)
#epsilon_abs_0 = 6.2e-2
#epsilon_abs_1 = 0.0
#epsilon_abs_2 = 2.0e-7

epsilon_abs_0 = 6.2e-2
epsilon_abs_1 = 0.0
epsilon_abs_2 = 2.0e-7

epsilon_sec_f_0 = 7.56e-1 
epsilon_sec_f_1 = -1.35e-3 
epsilon_sec_f_2 = 8.75e-7

epsilon_sec_b_0 = 0.99999999 
epsilon_sec_b_1 = 0 
epsilon_sec_b_2 = 0

epsilon_case = 0.57 
k_ins_0 = 4.85e-2 
k_ins_1 = -3.77e-5 
k_ins_2 = 4.8e-7   
k_sec = 220


# End of physical properties of materials ###############################################



# End of intermediate calculations


# String conversion

r2Str = str(r2)
r3Str = str(r3)
r4Str = str(r4)
r5Str = str(r5)
r6Str = str(r6)
s6_7Str = str(s6_7)
s8_9Str = str(s8_9)
r9Str = str(r9)
l_tubeStr = str(l_tube)
thickTubeStr = str(thickTube)
mode_evacStr = str(mode_evac)
mode_calcStr = str(mode_calc)

s_plateStr = str(s_plate)
b_plateStr = str(b_plate)
area_corStr = str(area_cor)


T_ambStr = str(T_amb)
v_windStr = str(v_wind)
T_initStr = str(T_init)
#T_endStr = str(T_end)


epsilon_abs_0Str = str(epsilon_abs_0)
epsilon_abs_1Str = str(epsilon_abs_1)
epsilon_abs_2Str = str(epsilon_abs_2)

epsilon_sec_f_0Str = str(epsilon_sec_f_0)
epsilon_sec_f_1Str = str(epsilon_sec_f_1)
epsilon_sec_f_2Str = str(epsilon_sec_f_2)

epsilon_sec_b_0Str = str(epsilon_sec_b_0)
epsilon_sec_b_1Str = str(epsilon_sec_b_1)
epsilon_sec_b_2Str = str(epsilon_sec_b_2)

epsilon_caseStr = str(epsilon_case)
k_ins_0Str = str(k_ins_0)
k_ins_1Str =  str(k_ins_1)
k_ins_2Str =    str(k_ins_2)
k_secStr = str(k_sec)


mode_calcStr = str(mode_calc)
mode_dpStr = str(mode_dp)

ident_htfStr = str(ident_htf)
mdot_htfStr = str(mdot_htf)

q7g_SolAbsStr = str(q7g_SolAbs)
q9_SolAbsStr = str(q9_SolAbs)
q6_SolAbsStr = str(q6_SolAbs)

p_htfStr = str(p_htf) 

# End of string conversion




os.system("./receiver_fcuevas"    +      " -r2 " + r2Str + \
                                         " -r3 " + r3Str + \
                                         " -r4 " + r4Str + \
                                         " -r5 " + r5Str + \
                                         " -r6 " + r6Str + \
                                         " -s6_7 " + s6_7Str + \
                                         " -s8_9 " + s8_9Str + \
                                         " -r9 " + r9Str + \
                                         " -l_tube " + l_tubeStr + \
                                         " -s_plate " + s_plateStr + \
                                         " -b_plate " + b_plateStr + \
                                         " -area_cor " + area_corStr + \
                                         " -mode_evac " + mode_evacStr + \
                                         
                                         " -T10 " + T_ambStr + \
                                         " -v_wind " + v_windStr + \
                                         
                                         " -mode_calc " + mode_calcStr + \
                                         " -mode_dp " + mode_dpStr + \
                                         " -T_init " + T_initStr + \
                                         " -p_htf " + p_htfStr + \
                                         " -mdot_htf " + mdot_htfStr + \
                                         " -ident_htf " + ident_htfStr + \
                                        
                                         " -epsilon_abs_0 " + epsilon_abs_0Str + \
                                         " -epsilon_abs_1 " + epsilon_abs_1Str + \
                                         " -epsilon_abs_2 " + epsilon_abs_2Str + \
                                         
                                         " -epsilon_sec_f_0 " + epsilon_sec_f_0Str + \
                                         " -epsilon_sec_f_1 " + epsilon_sec_f_1Str + \
                                         " -epsilon_sec_f_2 " + epsilon_sec_f_2Str + \
                                         
                                         " -epsilon_sec_b_0 " + epsilon_sec_b_0Str + \
                                         " -epsilon_sec_b_1 " + epsilon_sec_b_1Str + \
                                         " -epsilon_sec_b_2 " + epsilon_sec_b_2Str + \
                                         
                                         " -epsilon_case " + epsilon_caseStr + \
                                         " -k_ins_0 " + k_ins_0Str + \
                                         " -k_ins_1 " + k_ins_1Str + \
                                         " -k_ins_2 " + k_ins_2Str + \
                                         " -k_sec " + k_secStr )
                                         
                                                                           

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



# ------------------------------------------------------------------------------
# now we want to use certain steps, most likely of the absorber temperature t3,
# to interpolate the other columns to those steps and save everything to a file
# ------------------------------------------------------------------------------  
step_incr  = 50

# at first, all the entries of the reference column have to be determined that
# are supposed to be used for interpolation later on. the list step_index
# will contain all necessary rows.
i = 0
step = 0
step_index = []

totLength = results.size

# the step value must be higher than the minimum value
while step < min(results[reference_column]):
  step = step + step_incr
# end of while-loop

step_start = step

while (i < len(results)-1) and (step < max(results[reference_column])):
  while ((step-results[reference_column][i])*(step-results[reference_column][i+1])) > 0:
    i=i+1
    # end of while-loop
    # write values of determined rows to the index list
  step_index.append(i)
  step_index.append(i+1)  
  step = step+step_incr

# end of while-loop
#now the size of the array that can contain all of the selected results is known


selection_rows  = len(step_index)/2
selection_clmns = selected_columns.size
selection = [[0 for i in range(selection_clmns)] for j in range(selection_rows)]
# the result is a matrix filled with zeroes (for initialization)
# now the interpolation can be performed on any desired column


column = 0
for column_name in selected_columns:
  i = 0
  row = 0
  step = step_start
  while (i < len(step_index)):
    # read values for linear interpolation
    x0 = results[reference_column][step_index[i]]
    x  = step
    x1 = results[reference_column][step_index[i+1]]

    f0 = results[column_name][step_index[i]]
    f1 = results[column_name][step_index[i+1]]
    
    # perform linear interpolation
    f = f0 + (f1-f0)/(x1-x0)*(x-x0)
    
    # write calculated solutions to the selection array
    selection[row][column] = f
    
    #if (column==2):
      #print step, selection[row][column-1], selection[row][column]
      #stepStr=str(step) + "  " + str(selection[row][column-1]) + "  " + str(selection[row][column])
      #res_file.write(stepStr)
      #res_file.write( '\n')
 #   print selection[row][column]
    i    = i+2
    row  = row+1
    step = step+step_incr
    
    
    
  # end of while-loop
  column = column+1
  
#  print selection[row-1][1]

# end of while-loop

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


# copy your selected columns and interpolated values to a new file
interpol_file   = open('resultsTRM/'+me+'_rabs_interpol.txt', 'w')

#print selection
# print the header for the new file first
savetxt(interpol_file,np.column_stack(selected_columns),fmt='%12s')
# and then the rest of the data
savetxt(interpol_file,selection,fmt='%12.6f')

print "python: interpolated columns copied to",interpol_file.name, "!"

copy_file.close()
result_file.close()
interpol_file.close()
# repeat complete procedure for every command







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


coeff_file   = open('HL_results.dat', 'w')

coeff_file.write("Total length: " + str(totLength) + " (m)\n")
coeff_file.write("\nTotal heat loss calculated with TRM\n HL: " + str(tot_HL/1000) + " (kW)\n")

coeff_file.write("\nWall absorber temperatures\n")
coeff_file.write("\nT initial: " + str(tempInit) + " (°C)\n")
coeff_file.write("T end: " + str(tempEnd) + " (°C)\n")

coeff_file.write("\n*************************************************************************************************\n")

coeff_file.write("\nCurves fitting:\n")

coeff_file.write("\n*************************************************************************************************\n")
coeff_file.write("Fitting, curve (u0*DT + u1*DT^2), DT: Tabs - Tamb\n")
coeff_file.write("\n*************************************************************************************************\n")

coeff_file.write("\n")
coeff_file.write("Correlation coefficients\n")
#coeff_file.write(" -q_HL_0 "+str(p1[0])+" -q_HL_1 "+str(p1[1])+" -q_HL_2 "+str(p1[2])+" -q_HL_3 "+str(p1[3])+" -q_HL_4 "+str(p1[4]))
coeff_file.write("\n-u0 "+str(p1[0])+" -u1 "+str(p1[1]) + "\n")
coeff_file.write(" \nSquares difference (W/m): " + str(sqDiff))
coeff_file.write("\nSum squares differences (kW): " + str(sqDiffTot/1000) + "\n")

coeff_file.write("\nHeat loss ColSim correlations\n")
coeff_file.write("\nHeat loss model 2 (DT), HL: " + str(HL_corrModel2DT/1000) + "(kW)\n")
coeff_file.write("Heat loss model 2 (T), HL: " + str(HL_corrModel2/1000) + "(kW)\n")

coeff_file.write("\nHeat loss model 3 (DT), HL: " + str(HL_corrModel3DT/1000) + "(kW)\n")
coeff_file.write("Heat loss model 3 (T), HL: " + str(HL_corrModel3/1000) + "(kW)\n")

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


coeff_file.write("\n*************************************************************************************************\n")
coeff_file.write("\nFitting, curve (u0*DT + u1*DT^4), DT: Tabs - Tamb\n")
coeff_file.write("\n*************************************************************************************************\n")
  
coeff_file.write("\nCorrelation coefficients\n")

coeff_file.write("-u0 "+str(p1[0])+" -u1 "+str(p1[1]) + "\n")
coeff_file.write(" \nSquares difference (W/m): " + str(sqDiff))
coeff_file.write("\nSum squares differences (kW): " + str(sqDiffTot/1000) + "\n")


coeff_file.write("\nHeat loss ColSim correlations\n")
coeff_file.write("\nHeat loss model 1 (DT), HL: " + str(HL_corrModel1DT/1000) + "(kW)\n")
coeff_file.write("Heat loss model 1 (T), HL: " + str(HL_corrModel1/1000) + "(kW)\n")

coeff_file.write("\nHeat loss model 4 (DT), HL: " + str(HL_corrModel4DT/1000) + "(kW)\n")
coeff_file.write("Heat loss model 4 (T), HL: " + str(HL_corrModel4/1000) + "(kW)")
# end of while-loop  
      
############################################################################################




#variable to plot

x_axisPlotName = 'node'
y1_axisPlotName = 'q_HL_int'

x_axisPlot = results[x_axisPlotName]
plt.xlabel('Length (m)')
y1_axisPlot = results[y1_axisPlotName]	  
plt.ylabel('Heat loss (W/m)')

y2_axisPlot = fitEval1
y3_axisPlot = fitEval2

plt.plot(x_axisPlot,y1_axisPlot)

plt.plot(x_axisPlot,y2_axisPlot)

#plt.plot(grad[:,0:],grad[:,1:])

plt.plot(x_axisPlot,y3_axisPlot)



#plt.show()  


