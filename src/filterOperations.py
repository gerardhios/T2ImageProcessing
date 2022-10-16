from pathlib import Path
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

#Functions

# list of coordinates of the neigborhood
def neighborhoodCoordinates(x:int, y:int, neigborhood_size:int):
    neigborhood_coordinates = []
    N = (neigborhood_size - 1) // 2
    xp = list(range(-N, N + 1))
    yp = list(range(-N, N + 1))
    for i in xp:
        for j in yp:
            neigborhood_coordinates.append((x + i, y + j))
    return neigborhood_coordinates

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
def averageFilter(img: np.ndarray, kernel_size:int):
    length, width = img.shape
    zp = np.copy(img)
    rb = np.copy(img)
    peb = np.copy(img)
    for i in range(length):
        for j in range(width):
            neigborhood_coordinates = neighborhoodCoordinates(i, j, kernel_size)
            # Zero padding avarage filter
            zp[i, j] = zeroPadding(img, neigborhood_coordinates)
            
            # Replicated borders avarage filter
            rb[i, j] = replicatedBorders(img, neigborhood_coordinates)
            
            # Periodically extended borders avarage filter
            peb[i, j] = periodicallyExtendedBorders(img, neigborhood_coordinates)    
            
    # Show Images Result
    fig = plt.figure(figsize=(15, 5))
    plt.title('Average Filter')
    plt.axis('off')
    fig.add_subplot(1, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title('Original Image')
    
    fig.add_subplot(1, 4, 2)
    plt.imshow(zp, cmap='gray')
    plt.axis('off')
    plt.title('Zero Padding')
    
    fig.add_subplot(1, 4, 3)
    plt.imshow(rb, cmap='gray')
    plt.axis('off')
    plt.title('Repliacted Borders')
    
    fig.add_subplot(1, 4, 4)
    plt.imshow(peb, cmap='gray')
    plt.axis('off')
    plt.title('Periodically Extended Borders')
    
    fig.show()

# Gaussian filter
def gaussianFilter(img: np.ndarray, sigma:float):
    esb = np.zeros_like(img)
    epb = np.zeros_like(img)
    zp = np.zeros_like(img)
    
    esb = gaussian_filter(img, sigma, mode='reflect')
    epb = gaussian_filter(img, sigma, mode='wrap')
    zp = gaussian_filter(img, sigma, mode='constant', cval=0)
    
    # Show Images Result
    fig = plt.figure(figsize=(15, 5))
    plt.title('Gaussian Filter')
    plt.axis('off')
    fig.add_subplot(1, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title('Original Image')
    
    fig.add_subplot(1, 4, 2)
    plt.imshow(esb, cmap='gray')
    plt.axis('off')
    plt.title('Extended Symetric Borders')
    
    fig.add_subplot(1, 4, 3)
    plt.imshow(epb, cmap='gray')
    plt.axis('off')
    plt.title('Periodically Extended Borders')
    
    fig.add_subplot(1, 4, 4)
    plt.imshow(zp, cmap='gray')
    plt.axis('off')
    plt.title('Zero Padding')
    
    fig.show()
    
# Laplacian filter
def laplaceFilter(img:np.ndarray, kernel:np.ndarray):
    laplace = cv.filter2D(img, -1, kernel)
    
    # Show Images Result
    fig = plt.figure(figsize=(15, 5))
    plt.title('Laplace Filter')
    plt.axis('off')
    fig.add_subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title('Original Image')
    
    fig.add_subplot(1, 2, 2)
    plt.imshow(laplace, cmap='gray')
    plt.axis('off')
    plt.title('Laplace Filter Aplied')
    
    fig.show()
    input("Press Enter to continue...")
    
if __name__ == "__main__":
    image_path = Path("img/68.png")
    img = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
    averageFilter(img, 3)
    gaussianFilter(img, 0.5)
    laplaceFilter(img, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]))
    laplaceFilter(img, np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]]))