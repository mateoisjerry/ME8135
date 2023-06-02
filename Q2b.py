# Matthew Lisondra
# ME8135 Assignment 2 Q2b

# Import the necessary modules
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.stats import multivariate_normal
import math
import pygame
from pygame.locals import *

# -----------------------------------------------------------------------------------------------------

def convertToPolar(xytheta):
    currentXYTHETA = np.copy([xytheta[0]*1000,xytheta[1]*1000,xytheta[2]])
    
    rho = math.dist(currentXYTHETA[:2], CENTER)
    theta = np.arctan2((currentXYTHETA[1]-CENTER[1]),(currentXYTHETA[0]-CENTER[0])) - currentXYTHETA[2]
    
    polarx = np.array([rho, theta])
    
    return polarx

d=10. # OFFSET FROM CENTER STARTING 270 DEG COUNTER CLOCKWISE
CENTER = np.array([640.,360.])
GTx = np.array([CENTER[0]/1000., CENTER[1]/1000. + (d*10/1000. + d*5/1000.), 0.])
GTxPOLAR = convertToPolar(GTx)
Px = np.zeros((3,3))
Zx = np.array([0.,0.])
KFx = np.copy(GTx)
KFxPOLAR = np.copy(GTxPOLAR)

T = 1/8 # 1/8 seconds
r = 0.1 # in m
L = 0.3 #  L is the distance between the wheel, known as wheelbase, and is 0.3m.

Gt = np.identity(3)
transpose_Gt = np.transpose(Gt)

wsigmax2 = 0.01
wsigmay2 = 0.1
Rt = np.array([[wsigmax2,0.,0.],[0.,wsigmax2,0.],[0.,0.,wsigmay2]])*T # no need to square again

rsigmar2 = 0.1
rsigmab2 = 0.01
Qt = np.array([[rsigmar2,0],[0,rsigmab2]]) # no need to square again

def GT():
    global GTx
    global GTxPOLAR
    
    omegaw = np.random.normal(0,0.1) / 10. # scale
    omegaphi = np.random.normal(0,0.01) / 10. # scale
    epsilonvect = np.array([T*omegaw,T*omegaw,T*omegaphi])
    
    ur = 0.5
    ul = 0.5
    
    currentXY = np.copy([GTx[0]*1000,GTx[1]*1000])
    if math.dist(currentXY, CENTER) < d*10:
        ur = 1.
        ul = 0.1
    if math.dist(currentXY, CENTER) > (d+1)*10:
        ur = 0.1
        ul = 1.
        
    # print(math.dist(currentXY, CENTER))
    ut = np.array([(ur+ul)/2,(ur-ul)])
    
    THETAM = np.array([[T*r*np.cos(GTx[2]),0.],[T*r*np.sin(GTx[2]),0.],[0.,T*r/L]])
    GTxnew = np.matmul(Gt, GTx) + np.matmul(THETAM, ut) + epsilonvect # noise is the epsilonvect
    GTx = GTxnew
    # print(GTx)
    
    GTxPOLARnew = convertToPolar(GTx)
    GTxPOLAR = GTxPOLARnew
    # print(GTxPOLAR)

def KF_prediction():
    global Px
    
    firstP = np.matmul(Gt, Px)
    P_bar = np.matmul(firstP, transpose_Gt) + Rt
    Px = P_bar
    # print(Px)

def zt():
    global GTx
    global Zx
    rr = np.random.normal(0, rsigmar2) # range measurement noise
    rb = np.random.normal(0, rsigmab2) # bearing measurement noise
    delt = np.array([rr,rb])
    Zxnew = convertToPolar(GTx) + delt
    Zx = Zxnew
    # print(Zxnew)

def KF_measurement():
    global GTx
    global GTxPOLAR
    global Px
    global Zx
    global KFx
    
    A = (GTx[0] - (CENTER[0] / 1000.)) / GTxPOLAR[0]
    B = (GTx[1] - (CENTER[1] / 1000.)) / GTxPOLAR[0]
    C = 0 - (GTx[1] - (CENTER[1] / 1000.)) / (GTxPOLAR[0]**2)
    D = 0 - (GTx[0] - (CENTER[0] / 1000.)) / (GTxPOLAR[0]**2)
    
    Ht = np.array([[A,B,0.],[C,D,-1.]])
    transpose_Ht = np.transpose(Ht)
    # print(Ht)
    
    part1INV = np.matmul(Ht, Px)
    part2INV = np.matmul(part1INV, transpose_Ht) + Qt
    invPart = np.linalg.inv(part2INV)
    
    part1Kt = np.matmul(Px, transpose_Ht)
    Kt = np.matmul(part1Kt, invPart)
    
    muPart = Zx - Ht.dot(GTx)
    print(Ht.dot(GTx))
    munew = GTx + np.matmul(Kt, muPart)
    KFx = munew
    # print(KFx)
    
    PPart = np.identity(3) - np.matmul(Kt, Ht)
    Pnew = np.matmul(PPart, Px)
    Px = Pnew
    # print(Px)

# -----------------------------------------------------------------------------------------------------

pygame.init()
display_width = 1280
display_height = 720
gameDisplay = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption('Matthew Lisondra ME8135 Assignment 2 Q2b')

black = (0,0,0)
white = (255,255,255)

clock = pygame.time.Clock()
crashed = False

GTpoints=[]
GTpoints.append([GTx[0]*1000,GTx[1]*1000])

KFpoints=[]
KFpoints.append([KFx[0]*1000,KFx[1]*1000])

cov_width=[]
cov_hight=[]

imp = pygame.image.load("INFO-TEXT2.png").convert()

t = 0
while not crashed:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            crashed = True

    gameDisplay.fill(white)
    WHITE=(255,255,255)
    BLUE=(0,0,255)
    RED=(255,0,0)
    GREEN=(0,255,0)
    PINK = (255,192,203)

    GT()
    
    red = (180, 50, 50)
    size = (GTx[0]*1000-(Px[0,0]*1000)/2, GTx[1]*1000-(1000*Px[1,1])/2, Px[0,0]*1000, 1000*Px[1,1])
    pygame.draw.ellipse(gameDisplay, red, size,1)
    
    KF_prediction()
    
    gameDisplay.blit(imp, (600, 0))
    pygame.display.flip()

    
    if (t%8)==0:
        zt()
        KF_measurement()
        pygame.draw.circle(gameDisplay, PINK, ([KFx[0]*1000,KFx[1]*1000]),10)
        
    # KF DRAWING
    pygame.draw.polygon(gameDisplay, GREEN,
                        [[KFx[0]*1000,KFx[1]*1000],[KFx[0]*1000,KFx[1]*1000] ,
                        [KFx[0]*1000,KFx[1]*1000]])
    KFpoints.append([KFx[0]*1000,KFx[1]*1000])
    pygame.draw.lines(gameDisplay,GREEN,False,KFpoints,5)
    
    # GT DRAWING
    pygame.draw.polygon(gameDisplay, BLUE,
                        [[GTx[0]*1000,GTx[1]*1000],[GTx[0]*1000,GTx[1]*1000] ,
                        [GTx[0]*1000,GTx[1]*1000]])
    GTpoints.append([GTx[0]*1000,GTx[1]*1000])
    pygame.draw.lines(gameDisplay,BLUE,False,GTpoints,5)
    
    # PREDICTION DRAWING
    # pygame.draw.polygon(gameDisplay, RED,
    #                     [[KF_PREDx[0,0]*1000+50,KF_PREDx[1,0]*1000+50],[KF_PREDx[0,0]*1000+40,KF_PREDx[1,0]*1000+35] ,
    #                     [KF_PREDx[0,0]*1000+40,KF_PREDx[1,0]*1000+65]])
    # KF_PREDpoints.append([KF_PREDx[0,0]*1000+50,KF_PREDx[1,0]*1000+50])
    # pygame.draw.lines(gameDisplay,RED,False,KF_PREDpoints,5)
    
    
    cov_hight.append(Px[0,0])
    cov_width.append(Px[1,1])
    pygame.draw.circle(gameDisplay, RED, (CENTER[0],CENTER[1]),5)
    
    pygame.display.update()
    clock.tick(8) 
        
    t+=1