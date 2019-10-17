import numpy as np
import cv2
import skimage as sk

from matplotlib import pyplot as plt

#Calcul des dériées en x et en y d'une gaussienne bi-dimensionnelle
def dGx(x,y,sigma):
    return((-x*np.exp(-(x**2+y**2)/(2*sigma**2))/(2*np.pi*sigma**4)))
def dGy(x,y,sigma):
    return((-y*np.exp(-(x**2+y**2)/(2*sigma**2))/(2*np.pi*sigma**4)))

#Construction de la dérivée selon x d'un noyau gaussien
def buildKernelX(sigma,size):
    assert(size%2==1)
    kernel = np.empty((size,size))
    middle = size//2

    for i in range(size):
        for j in range(size):
            kernel[i,j] = dGx(j-middle,i-middle,sigma)

    return(kernel)

#Construction de la dérivée selon y d'un noyau gaussien
def buildKernelY(sigma,size):
    assert(size%2==1)
    kernel = np.empty((size,size))
    middle = size//2

    for i in range(size):
        for j in range(size):
            kernel[i,j] = dGy(j-middle,i-middle,sigma)
    return(kernel)

#Lecture image en niveau de gris et conversion en float64
img = np.float64(cv2.imread('Image_Pairs/Graffiti0.png',cv2.IMREAD_GRAYSCALE))
(h,w) = img.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes")
print("Type de l'image :",img.dtype)

#Paramètres du calcul
sigma1 = 1
size1 = 3
size2 = 3
sigma2 = 2*sigma1
alpha = 0.06

#Début du calcul
t1 = cv2.getTickCount()
Theta = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)

# Calcul de la fonction d'intérêt de Harris

#Construction des kernels gaussiens dérivés
kernelx = buildKernelX(sigma1,size1)
kernely = buildKernelY(sigma1,size1)

#Calcul des dérivées de l'image à l'échelle sigma1
Ix = cv2.filter2D(img,-1,kernelx)
Iy = cv2.filter2D(img,-1,kernely)

#Calcul des valeurs de la matrice d'auto-corrélation
Ix2 = Ix**2
Iy2 = Iy**2
IxIy = np.multiply(Ix,Iy)

#Moyennage de la matrice d'auto-corrélation
Ix2 = cv2.GaussianBlur(Ix2,(size2,size2),sigmaX=sigma2,sigmaY=sigma2)
Iy2 = cv2.GaussianBlur(Iy2,(size2,size2),sigmaX=sigma2,sigmaY=sigma2)
IxIy = cv2.GaussianBlur(IxIy,(size2,size2),sigmaX=sigma2,sigmaY=sigma2)

#Création de l'image contenant la fonction d'interet de Harris
Theta = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)

#Calcul de la fonction d'interet de Harris
for y in range(1,h-1):
    for x in range(1,w-1):
        mat = np.array([[Ix2[y,x],IxIy[y,x]],[IxIy[y,x],Iy2[y,x]]])
        val = np.linalg.det(mat) - alpha*(np.trace(mat)**2)
        Theta[y,x] = val

# Calcul des maxima locaux et seuillage
Theta_maxloc = cv2.copyMakeBorder(Theta,0,0,0,0,cv2.BORDER_REPLICATE)
d_maxloc = 3
seuil_relatif = 0.01
se = np.ones((d_maxloc,d_maxloc),np.uint8)
Theta_dil = cv2.dilate(Theta,se)
#Suppression des non-maxima-locaux
Theta_maxloc[Theta < Theta_dil] = 0.0
#On néglige également les valeurs trop faibles
Theta_maxloc[Theta < seuil_relatif*Theta.max()] = 0.0
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Mon calcul des points de Harris :",time,"s")
print("Nombre de cycles par pixel :",(t2 - t1)/(h*w),"cpp")

fig1 = plt.figure()
fig1.suptitle('Implementation "à la main"')

plt.subplot(131)
plt.imshow(img,cmap = 'gray')
plt.title('Image originale')

plt.subplot(132)
plt.imshow(Theta,cmap = 'gray')
plt.title('Fonction de Harris')

se_croix = np.uint8([[1, 0, 0, 0, 1],
[0, 1, 0, 1, 0],[0, 0, 1, 0, 0],
[0, 1, 0, 1, 0],[1, 0, 0, 0, 1]])
Theta_ml_dil = cv2.dilate(Theta_maxloc,se_croix)
#Relecture image pour affichage couleur
Img_pts=cv2.imread('Image_Pairs/Graffiti0.png',cv2.IMREAD_COLOR)
(h,w,c) = Img_pts.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes x",c,"canaux")
print("Type de l'image :",Img_pts.dtype)
#On affiche les points (croix) en rouge
Img_pts[Theta_ml_dil > 0] = [255,0,0]
plt.subplot(133)
plt.imshow(Img_pts)
plt.title('Points de Harris')

#Comparaison avec la fonction cv2.cornerHarris

#Début du calcul
t1 = cv2.getTickCount()

img = np.uint8(img)
Theta = cv2.cornerHarris(img,size2,size1,alpha)

# Calcul des maxima locaux et seuillage
Theta_maxloc = cv2.copyMakeBorder(Theta,0,0,0,0,cv2.BORDER_REPLICATE)
se = np.ones((d_maxloc,d_maxloc),np.uint8)
Theta_dil = cv2.dilate(Theta,se)
#Suppression des non-maxima-locaux
Theta_maxloc[Theta < Theta_dil] = 0.0
#On néglige également les valeurs trop faibles
Theta_maxloc[Theta < seuil_relatif*Theta.max()] = 0.0
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Calcul des points de Harris OpenCV:",time,"s")
print("Nombre de cycles par pixel :",(t2 - t1)/(h*w),"cpp")

fig2 = plt.figure()
fig2.suptitle("Implementation OpenCV")

plt.subplot(131)
plt.imshow(img,cmap = 'gray')
plt.title('Image originale')

plt.subplot(132)
plt.imshow(Theta,cmap = 'gray')
plt.title('Fonction de Harris')

Theta_ml_dil = cv2.dilate(Theta_maxloc,se_croix)

#On affiche les points (croix) en rouge
Img_pts[Theta_ml_dil > 0] = [255,0,0]
plt.subplot(133)
plt.imshow(Img_pts)
plt.title('Points de Harris')

plt.show()

