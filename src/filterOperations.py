from pathlib import Path
import numpy as np
import cv2 as cv

#Functions

# Zero padding
def zeroPadding(img:np.ndarray, neigborhood_coordinates:list):
    xmax = img.shape[0] - 1
    ymax = img.shape[1] - 1
    neigborhood_values = []
    for coordinate in neigborhood_coordinates:
        if coordinate[0] < 0 or coordinate[1] < 0:
            neigborhood_values.append(0)
        elif coordinate[0] > xmax or coordinate[1] > ymax:
            neigborhood_values.append(0)
        else:
            neigborhood_values.append(img[coordinate[0], coordinate[1]])
    return sum(neigborhood_values) // len(neigborhood_values)

# Replicated borders
def replicatedBorders(img:np.ndarray, neigborhood_coordinates:list):
    xmax = img.shape[0] - 1
    ymax = img.shape[1] - 1
    neigborhood_values = []
    for coordinate in neigborhood_coordinates:
        ####Esquinas####
        # Esquina superior izquierda
        if coordinate[0] < 0 and coordinate[1] < 0:
            neigborhood_values.append(img[0, 0])
            
        # Esquina superior derecha
        elif coordinate[0] > xmax and coordinate[1] < 0:
            neigborhood_values.append(img[xmax, 0])
            
        # Esquina inferior izquierda
        elif coordinate[0] < 0 and coordinate[1] > ymax:
            neigborhood_values.append(img[0, ymax])
        
        # Esquina inferior derecha
        elif coordinate[0] > xmax and coordinate[1] > ymax:
            neigborhood_values.append(img[xmax, ymax])
        
        ####Bordes####
        # Borde superior
        elif coordinate[1] < 0 and coordinate[0] >= 0 and coordinate[0] <= xmax:
            neigborhood_values.append(img[coordinate[0], 0])
        
        # Borde inferior
        elif coordinate[1] > ymax and coordinate[0] >= 0 and coordinate[0] <= xmax:
            neigborhood_values.append(img[coordinate[0], ymax])
        
        # Borde izquierdo
        elif coordinate[0] < 0 and coordinate[1] >= 0 and coordinate[1] <= ymax:
            neigborhood_values.append(img[0, coordinate[1]])
        
        # Borde derecho
        elif coordinate[0] > xmax and coordinate[1] >= 0 and coordinate[1] <= ymax:
            neigborhood_values.append(img[xmax, coordinate[1]])
            
        # Centro
        else:
            neigborhood_values.append(img[coordinate[0], coordinate[1]])   
                             
    return sum(neigborhood_values) // len(neigborhood_values)

# Periodically extended borders
def periodicallyExtendedBorders(img:np.ndarray, neigborhood_coordinates:list):
    xmax = img.shape[0] - 1
    ymax = img.shape[1] - 1
    neigborhood_values = []
    for coordinate in neigborhood_coordinates:
        ####Esquinas####
        # Esquina inferior derecha
        if coordinate[0] > xmax and coordinate[1] > ymax:
            neigborhood_values.append(img[0, 0])
            
        ####Bordes####
        # Borde derecho
        elif coordinate[0] > xmax and coordinate[1] <= ymax:
            neigborhood_values.append(img[0, coordinate[1]])
            
        # Borde inferior
        elif coordinate[1] > ymax and coordinate[0] <= xmax:
            neigborhood_values.append(img[coordinate[0], 0])
            
        else:
            neigborhood_values.append(img[coordinate[0], coordinate[1]])
            
    return sum(neigborhood_values) // len(neigborhood_values)
    
# average image filter
def averageFilter(img: np.ndarray):
    length, width = img.shape
    
    zp = np.copy(img)
    rb = np.copy(img)
    peb = np.copy(img)
    for i in range(length):
        for j in range(width):
            neigborhood_coordinates = [
                (i - 1, j - 1), (i, j - 1), (i + 1, j - 1),
                (i - 1,   j  ), (i,   j  ), (i + 1,   j  ),
                (i - 1, j + 1), (i, j + 1), (i + 1, j + 1),
            ]
            # Zero padding avarage filter
            zp[i, j] = zeroPadding(img, neigborhood_coordinates)
            
            # Replicated borders avarage filter
            rb[i, j] = replicatedBorders(img, neigborhood_coordinates)
            
            # Periodically extended borders avarage filter
            peb[i, j] = periodicallyExtendedBorders(img, neigborhood_coordinates)    
            
    # Show Images Result
    cv.imshow('Original', img)
    cv.imshow('A.F. Zero Padding', zp)
    cv.imshow('A.F. Replicated Borders', rb)
    cv.imshow('A.F. P. Extended Borders', peb)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    
if __name__ == "__main__":
    image_path = Path("img/68.png")
    img = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
    averageFilter(img)