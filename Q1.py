# Matthew Lisondra
# ME8135 Assignment 2 Q1

# Import the necessary modules
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.stats import multivariate_normal
import math
import pygame
from pygame.locals import *

# -----------------------------------------------------------------------------------------------------

GTx = np.array([0,0])
Px = np.zeros((2,2))
Zx = np.array([0,0])
KFx = np.array([0,0])

T = 1/8 # 1/8 seconds
r = 0.1 # in m

At = np.identity(2)
transpose_At = np.transpose(At)
Bt = np.zeros((2,2)) + (T*r/2)
Ct = np.array([[1.,0.],[0.,2.]]) # Has to be made float
transpose_Ct = np.transpose(Ct)

wsigmax2 = 0.1
wsigmay2 = 0.15
Rt = np.array([[wsigmax2,0],[0,wsigmay2]])*T # no need to square again

ut = np.array([1.,0.1]) # controls, we can make them anything

rsigmax2 = 0.05
rsigmay2 = 0.075
Qt = np.array([[rsigmax2,0],[0,rsigmay2]]) # no need to square again

def GT():
    global GTx
    omegax = np.random.normal(0,0.1)
    omegay = np.random.normal(0,0.15)
    epsilonvect = np.array([[omegax*T],[omegay*T]])
    GTxnew = np.matmul(At, GTx) + np.matmul(Bt, ut) + epsilonvect
    GTx = GTxnew
    # print((GTx[0,0],GTx[1,0]))

def KF_prediction():
    global Px
    firstP = np.matmul(At, Px)
    P_bar = np.matmul(firstP, transpose_At) + Rt
    Px = P_bar
    # print(Px)

def zt():
    global GTx
    global Zx
    rx = np.random.normal(0, rsigmax2)
    ry = np.random.normal(0, rsigmay2)
    delt = np.array([rx,ry])
    Zxnew = Ct.dot(GTx) + delt
    Zx = Zxnew
    print(Zxnew)

def KF_measurement():
    global GTx
    global Px
    global Zx
    global KFx
    part1INV = np.matmul(Ct, Px)
    part2INV = np.matmul(part1INV, transpose_Ct) + Qt
    invPart = np.linalg.inv(part2INV)
    
    part1Kt = np.matmul(Px, transpose_Ct)
    Kt = np.matmul(part1Kt, invPart)
    
    muPart = Zx - Ct.dot(GTx)
    munew = GTx + np.matmul(Kt, muPart)
    KFx = munew
    
    PPart = np.identity(2) - np.matmul(Kt, Ct)
    Pnew = np.matmul(PPart, Px)
    Px = Pnew

# -----------------------------------------------------------------------------------------------------

pygame.init()
display_width = 1280
display_height = 720
gameDisplay = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption('Matthew Lisondra ME8135 Assignment 2 Q1')

black = (0,0,0)
white = (255,255,255)

clock = pygame.time.Clock()
crashed = False

GTpoints=[]
GTpoints.append([0,0])

KFpoints=[]
KFpoints.append([0,0])

cov_width=[]
cov_hight=[]

imp = pygame.image.load("INFO-TEXT.png").convert()

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

    GT()
    
    red = (180, 50, 50)
    size = (GTx[0,0]*1000+50-(Px[0,0]*2000)/2, GTx[1,0]*1000+50-(2000*Px[1,1])/2, Px[0,0]*2000, 2000*Px[1,1])
    pygame.draw.ellipse(gameDisplay, red, size,1)  
    
    KF_prediction()
    
    gameDisplay.blit(imp, (600, 0))
    pygame.display.flip()

    
    if (t%8)==0:
        zt()
        KF_measurement()
        
    # KF DRAWING
    pygame.draw.polygon(gameDisplay, GREEN,
                        [[KFx[0,0]*1000+50,KFx[1,0]*1000+50],[KFx[0,0]*1000+40,KFx[1,0]*1000+35] ,
                        [KFx[0,0]*1000+40,KFx[1,0]*1000+65]])
    KFpoints.append([KFx[0,0]*1000+50,KFx[1,0]*1000+50])
    pygame.draw.lines(gameDisplay,GREEN,False,KFpoints,5)
    
    # GT DRAWING
    pygame.draw.polygon(gameDisplay, BLUE,
                        [[GTx[0,0]*1000+50,GTx[1,0]*1000+50],[GTx[0,0]*1000+40,GTx[1,0]*1000+35] ,
                        [GTx[0,0]*1000+40,GTx[1,0]*1000+65]])
    GTpoints.append([GTx[0,0]*1000+50,GTx[1,0]*1000+50])
    pygame.draw.lines(gameDisplay,BLUE,False,GTpoints,5)
    
    # PREDICTION DRAWING
    # pygame.draw.polygon(gameDisplay, RED,
    #                     [[KF_PREDx[0,0]*1000+50,KF_PREDx[1,0]*1000+50],[KF_PREDx[0,0]*1000+40,KF_PREDx[1,0]*1000+35] ,
    #                     [KF_PREDx[0,0]*1000+40,KF_PREDx[1,0]*1000+65]])
    # KF_PREDpoints.append([KF_PREDx[0,0]*1000+50,KF_PREDx[1,0]*1000+50])
    # pygame.draw.lines(gameDisplay,RED,False,KF_PREDpoints,5)
    
    
    cov_hight.append(Px[0,0])
    cov_width.append(Px[1,1])
    pygame.display.update()
    clock.tick(8) 
        
    t+=1