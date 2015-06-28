# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 16:50:15 2015

@author: fcuevas
"""

from numpy import genfromtxt
import numpy as np
import time

#theta = [0, 10, 20, 30, 40, 50, 60, 70, 80, 85]
#gamma = [0, 90]

theta = [0]
gamma = [90]

for gamma_s in gamma:
    
    for theta_z in theta:
        
        t1 = time.time()

        readFile = "rayoutTheta_z" + str(theta_z) + "azi_" + str(gamma_s) + ".dat"
        
        r3 = genfromtxt('rayoutTheta_z0_azi_90.dat', comments="#")
        #r3 = genfromtxt('rayoutTemp.dat', comments="#")
        
        mirrorFile = "mirrorContribution" + str(theta_z) + "azi_" + str(gamma_s) + ".dat"
        secFile = "secondaryContribution" + str(theta_z) + "azi_" + str(gamma_s) + ".dat"
        mCont = open(mirrorFile, "w")
        secCont = open(secFile, "w")
        totRes = open("resFile.dat", "w")
        
        #check the contribution of every mirror
        totRays = r3[-1,0]
        print totRays
        mirrors = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])
        
        secondary = range(21,532)
        contrSec = np.array([np.zeros(len(secondary)*4)]).reshape(len(secondary),4)
        
        contrMirror = np.array([np.zeros(len(mirrors)*7)]).reshape(len(mirrors),7)
        
        
        contrMirror[:,0] = mirrors
        contrSec[:,0] = secondary
        
        absorbedEnergy = True
        
        if absorbedEnergy:
            # 
            totEabs=0
            hitAbs = 0
            hitMirr=0
            for totRay in xrange(len(r3)):
                
                if r3[totRay,2] in mirrors:
                    hitMirr +=1
                
                if r3[totRay,2] == 539:  
                    Eabs = r3[totRay-1, 6] - r3[totRay, 6] 
                    totEabs = totEabs + Eabs
                    hitAbs +=1
            
            intercept = hitAbs/totRays 
            interceptReal = hitAbs/hitMirr     
            resGlob = [totEabs/totRays, totEabs/hitMirr, hitAbs, hitMirr, intercept, interceptReal]
#            np.savetxt(mCont, resGlob) 
#                       fmt=['%12.5f','%12.5f','%12d','%12d','%12.5f','%12.5f'])
            print totEabs/totRays, totEabs/hitMirr, hitAbs, hitMirr, intercept, interceptReal
            t2 = time.time()
            
            timeC1 = t2-t1
        
            print "time", timeC1
        
        
        mirrorContribution = True
        
        if mirrorContribution:
            # 
            hitMirr=0
            for totRay in xrange(len(r3)):
                
                if r3[totRay,2] in mirrors:       
                    hitMirr += 1
                    totHit = 0
                    totEabs = 0
                    mirrorID = r3[totRay,2]
                    totHitSec = 0
                    posID = mirrorID / 2
                    rayID = r3[totRay,0]        
                    
                    while r3[totRay,0] == rayID:
                        
                        totRay = totRay + 1
                        
                        if r3[totRay,2] in secondary:
                            totHitSec = totHitSec + 1
              
                        
                        if r3[totRay, 2] == 539:
                            totEabs = r3[totRay-1, 6] - r3[totRay, 6] 
                            
                            
                            contrMirror[posID, 1] = contrMirror[posID, 1] + 1
                            contrMirror[posID, 2] = contrMirror[posID, 2] + totEabs
                            contrMirror[posID, 6] += totHitSec
                            break
        
                        
                
            
            for totRay in xrange(len(r3)):
                
                if r3[totRay,2] in secondary:
                       
                    totHit = 0
                    totEabs = 0
                    secID = r3[totRay,2]
                    posID = secID - 21
                    rayID = r3[totRay,0]        
                    
                    while r3[totRay,0] == rayID:
            
                        totRay = totRay + 1
                        if r3[totRay, 2] == 539:
                            Eabs = r3[totRay-1, 6] - r3[totRay, 6] 
                            
                            
                            contrSec[posID, 1] = contrSec[posID, 1] + 1
                            contrSec[posID, 2] = contrSec[posID, 2] + Eabs
                            break
            
            
            
            for totRay in xrange(len(r3)):
                
                if r3[totRay,2] in mirrors:        
                    
                    totHit = 0
                    totEabs = 0
                    mirrorID = r3[totRay,2]
                    
                    
                    posID = mirrorID / 2
                    rayID = r3[totRay,0]        
                    
                    while r3[totRay,0] == rayID:
                        
                        totRay = totRay + 1
                        
                        if r3[totRay,0] in secondary:
                            break
            
            
                        elif r3[totRay, 2] == 539:
                            Eabs = r3[totRay-1, 6] - r3[totRay, 6] 
                            
                            
                            contrMirror[posID, 4] = contrMirror[posID, 4] + 1
                            contrMirror[posID, 5] = contrMirror[posID, 5] + Eabs
                            break
                        
            
            contrMirror[:,3] = contrMirror[:,2]/(hitMirr)                 
#            print contrMirror
#            print sum(contrMirror[:,3]), sum(contrMirror[:,1])
            
            np.savetxt(mCont,contrMirror, fmt=['%6d','%12d','%12.5f','%12.5f','%12d','%12.5f', '%12d'])
            np.savetxt(totRes,contrMirror, fmt=['%6d','%12d','%12.5f','%12.5f','%12d','%12.5f', '%12d'])
        
        #X, Y, Z = meshg
        
        filterByMirror = True
        
        if filterByMirror:
            
            mirrorDict = {}
            for mirror in mirrors:
                
                dataTemp = []
                key = "mirror" + str(mirror)
                
                for totRay in xrange(len(r3)):
                    
                    if r3[totRay,2] == mirror:
                        
                        dataTemp.append(r3[totRay,:])
                        
                mirrorDict[key] = dataTemp
                mCont.write("{0:^2} {1:^12} {0:^2}".format("\n",key, "\n"))
                np.savetxt(mCont,mirrorDict[key], fmt=['%6d','%6d','%6d','%12.5f','%12d','%12.5f', '%12.5f'])
                
                
        
                        
        hitSecundary = True
        
        
        if hitSecundary:
            hitSec = []
            for totRay in xrange(len(r3)):
                if r3[totRay,2] in secondary:
                    
                    Eabs = r3[totRay-1, 6] - r3[totRay, 6]
                    res = np.append(r3[totRay,:], Eabs)
                    hitSec.append(res)
                    
                    
            print np.shape(hitSec)
            hitSec = np.asarray(hitSec)
            
            freqLength, binsLength = np.histogram(hitSec[:,4], bins=50)     
            print freqLength, binsLength
            
        #    freqWidth, binsWidth = np.histogram(hitSec[:,4], bins=20, range=(-1.0, 1.0))     
        #    print freqWidth, binsWidth
            
            np.savetxt(secCont, freqLength, fmt=['%12.5f'])
            np.savetxt(secCont, binsLength, fmt=['%12.5f'])
            np.savetxt(secCont,hitSec, fmt=['%6d','%6d','%6d','%12.5f','%12.5f','%12.5f', '%12.5f', '%12.5f'])
            
            
#        mapSec = False
#        
#        if mapSec:
#            secDict = {}
#            for sec in secondary:
#                dataTemp = []
#                key = "sec" + str(sec)
#                
#                for totRay in xrange(len(r3)):
#                    
#                    if r3[totRay,2] == sec:
#                        
#                        dataTemp.append(r3[totRay,:])
#                        
#                secDict[key] = dataTemp
            
                
                
                
            
