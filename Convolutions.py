import numpy as np
import cv2

from matplotlib import pyplot as plt

#Creation des axes de plot
figContraste, axesContraste = plt.subplots(1, 2)
figContraste.suptitle("Rehaussement de contraste")
figIx, axesIx = plt.subplots(1, 2)
figIx.suptitle("Dérivée selon x")
figDeltaI, axesDeltaI = plt.subplots(1, 2)
figDeltaI.suptitle("Norme du gradient")

#Lecture image en niveau de gris et conversion en float64
img=np.float64(cv2.imread('Image_Pairs/FlowerGarden2.png',0))
(h,w) = img.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes")

#Methode directe
t1 = cv2.getTickCount()

#Creation des images
IContraste = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
Ix = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
DeltaI = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)

#Application du filtre de convolution
for y in range(1,h-1):
  for x in range(1,w-1):
    #Noyau de convolution pour le rehaussement de contraste
    val = 5*img[y, x] - img[y-1, x] - img[y, x-1] - img[y+1, x] - img[y, x+1] 
    #Noyau de convolution pour le calcul de la dérivée selon x
    valx = -img[y-1,x-1] + img[y-1,x+1] - 2*img[y,x-1] + 2*img[y,x+1] - img[y+1,x-1] + img[y+1,x-1] 
    #Noyau de convolution pour le calcul de la dérivée selon y
    valy = -img[y-1,x-1] - 2*img[y-1,x] - img[y-1,x+1] + img[y+1,x-1] +2*img[y+1,x] + img[y+1,x-1]

    #Saturation des valeurs 
    IContraste[y,x] = min(max(val,0),255) 
    Ix[y,x] = min(max(valx,0),255)
    DeltaI[y,x] = min(max(np.sqrt(valx**2 + valy**2),0),255)

t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Méthode directe :",time,"s")

axesContraste[0].imshow(IContraste,cmap = 'gray',vmin=0.0,vmax=255.0)
axesContraste[0].set_title('Convolution - Méthode Directe')

axesIx[0].imshow(Ix,cmap = 'gray',vmin=0.0,vmax=255.0)
axesIx[0].set_title('Convolution - Méthode Directe')

axesDeltaI[0].imshow(DeltaI,cmap = 'gray',vmin=0.0,vmax=255.0)
axesDeltaI[0].set_title('Convolution - Méthode Directe')

#Méthode filter2D
t1 = cv2.getTickCount()

#Noyaux de convolution
kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
kernelx = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
kernely = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])

#Application des filtres de convolution
IContraste = cv2.filter2D(img,-1,kernel)
Ix = cv2.filter2D(img,-1,kernelx)
DeltaI = np.sqrt(cv2.filter2D(img,-1,kernelx)**2 + cv2.filter2D(img,-1,kernely)**2)

t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Méthode filter2D :",time,"s")

axesContraste[1].imshow(IContraste,cmap = 'gray',vmin=0.0,vmax=255.0)
axesContraste[1].set_title('Convolution - filter2D')

axesIx[1].imshow(Ix,cmap = 'gray',vmin=0.0,vmax=255.0)
axesIx[1].set_title('Convolution - filter2D')

axesDeltaI[1].imshow(DeltaI,cmap = 'gray',vmin=0.0,vmax=255.0)
axesDeltaI[1].set_title('Convolution - filter2D')

plt.show()
