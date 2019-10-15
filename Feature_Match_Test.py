import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy.linalg as lg

def build_transform(image,rotation,dilatation):
    transform = cv2.getRotationMatrix2D((image.shape[0]/2,image.shape[1]/2),rotation,dilatation)
    image2 = cv2.warpAffine(image,transform, dsize = (image.shape[0],image.shape[1]))
    return(image2,transform)


def test_feature_match(pts_query,pts_train,matches,transform,detector):
    #Scale space and counters
    S = np.logspace(-3,1,50)
    C = np.zeros(len(S))

    #Total error
    E = 0

    #Transformation estimation
    X=np.zeros((len(matches),3))
    Y=np.zeros((len(matches),2))

    for i,m in enumerate(matches):
        try:
            m=m[0]
        except:
            pass
        
        img1_idx = m.queryIdx
        img2_idx = m.trainIdx

        (x1,y1) = pts_query[img1_idx].pt
        (x2,y2) = pts_train[img2_idx].pt

        x2c = transform[0,0]*x1 + transform[0,1]*y1 + transform[0,2]
        y2c = transform[1,0]*x1 + transform[1,1]*y1 + transform[1,2]

        #Compute error
        error = np.sqrt((x2c-x2)**2+(y2c-y2)**2)
        E+=error/len(matches)

        for j,treshold in enumerate(S):
            #If the keypoint is correct according to the scale
            if(error<treshold):
                C[j] += 100/len(matches)
    
        Y[i,:] = [x2,y2]
        X[i,:] = [x1,y1,1]
    
    print("\n -- TEST DE PERFORMANCE DU FEATURE MATCHING -- \n")
    #Total mean error
    print("Erreur totale moyenne = ",E," pixels")

    #Accuracy according to the scale
    plt.plot(S,C,label=detector)
    plt.xscale('log')
    plt.xlabel("Seuil de précision (en pixels)")
    plt.ylabel("Pourcentage de points d'intérêts corrects (en %)")
    plt.legend()
    plt.show()

    #Estimation of the transformation matrix using lstsq
    A = lg.lstsq(X,Y,rcond=None)[0]
    approx_transform = A.T
    error_transform = np.sqrt((transform-approx_transform)**2)
    print("Matrice d'erreur sur la transformation approximée :")
    print(error_transform)

    


            