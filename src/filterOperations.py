from pathlib import Path
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

#Functions

# list of coordinates of the neigborhood
def neighborhoodCoordinates(x:int, y:int, neigborhood_size:int, sorted:bool=False):
    neigborhood_coordinates = []
    N = (neigborhood_size - 1) // 2
    xp = list(range(-N, N + 1))
    yp = list(range(-N, N + 1))
    for i in xp:
        for j in yp:
            neigborhood_coordinates.append((x + i, y + j))
    if sorted:
        ret = []
        for i in range(neigborhood_size):
             ret.append([])
        cont = 0
        for i in range(neigborhood_size):
            for j in range(neigborhood_size):
                ret[j].append(neigborhood_coordinates[cont])
                cont += 1
        return ret
    return neigborhood_coordinates
    
# Gaussian Kernel
def gaussian_kernel(kernel_size:int, sigma:float) -> np.ndarray:
    coords = neighborhoodCoordinates(0, 0, kernel_size, sorted=True)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i, j] = np.exp(-(coords[i][j][0] ** 2 + coords[i][j][1] ** 2)/(2 * (sigma ** 2)))
    return kernel / kernel.sum()
 
####Border types####
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

def zeroPaddingKernel(img:np.ndarray, neigborhood_coordinates:list, kernel:np.ndarray):
    xmax = img.shape[0] - 1
    ymax = img.shape[1] - 1
    neigborhood_values = np.ones((kernel.shape[0], kernel.shape[1]), dtype=np.int16)
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            x = neigborhood_coordinates[i][j][0]
            y = neigborhood_coordinates[i][j][1]
            if x < 0 or y < 0:
                neigborhood_values[i, j] = 0
            elif x > xmax or y > ymax:
                neigborhood_values[i, j] = 0
            else:
                neigborhood_values[i, j] = img[x, y]
    return int(np.sum(neigborhood_values * kernel))

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

def replicatedBordersKernel(img:np.ndarray, neigborhood_coordinates:list, kernel:np.ndarray):
    xmax = img.shape[0] - 1
    ymax = img.shape[1] - 1
    neigborhood_values = np.ones((kernel.shape[0], kernel.shape[1]), dtype=np.int16)
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            x = neigborhood_coordinates[i][j][0]
            y = neigborhood_coordinates[i][j][1]
            ####Esquinas####
            # Esquina superior izquierda
            if x < 0 and y < 0:
                neigborhood_values[i,j] = img[0, 0]

            # Esquina superior derecha
            elif x > xmax and y < 0:
                neigborhood_values[i,j] = img[xmax, 0]

            # Esquina inferior izquierda
            elif x < 0 and y > ymax:
                neigborhood_values[i,j] = img[0, ymax]

            # Esquina inferior derecha
            elif x > xmax and y > ymax:
                neigborhood_values[i,j] = img[xmax, ymax]

            ####Bordes####
            # Borde superior
            elif y < 0 and x >= 0 and x <= xmax:
                neigborhood_values[i,j] = img[x, 0]

            # Borde inferior
            elif y > ymax and x >= 0 and x <= xmax:
                neigborhood_values[i,j] = img[x, ymax]

            # Borde izquierdo
            elif x < 0 and y >= 0 and y <= ymax:
                neigborhood_values[i,j] = img[0, y]

            # Borde derecho
            elif x > xmax and y >= 0 and y <= ymax:
                neigborhood_values[i,j] = img[xmax, y]

            # Centro
            else:
                neigborhood_values[i,j] = img[x, y]
                             
    return int(np.sum(neigborhood_values * kernel))

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

def periodicallyExtendedBordersKernel(img:np.ndarray, neigborhood_coordinates:list, kernel:np.ndarray):
    xmax = img.shape[0] - 1
    ymax = img.shape[1] - 1
    neigborhood_values = np.ones((kernel.shape[0], kernel.shape[1]), dtype=np.int16)
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            x = neigborhood_coordinates[i][j][0]
            y = neigborhood_coordinates[i][j][1]
            ####Esquinas####
            # Esquina inferior derecha
            if x > xmax and y > ymax:
                neigborhood_values[i, j] = img[0, 0]

            ####Bordes####
            # Borde derecho
            elif x > xmax and y <= ymax:
                neigborhood_values[i, j] = img[0, y]

            # Borde inferior
            elif y > ymax and x <= xmax:
                neigborhood_values[i, j] = img[x, 0]

            else:
                neigborhood_values[i, j] = img[x, y]
            
    return int(np.sum(neigborhood_values * kernel))

# Obtain the coordinates of the symmetric neighborhood
def symetricalcoordinate(coordinate:int, maxcoordinate:int):
    if coordinate == -1:
        return 0
    elif coordinate < -1:
        return -coordinate - 1
    elif coordinate == maxcoordinate + 1:
        return maxcoordinate
    elif coordinate > maxcoordinate + 1:
        return (2 * maxcoordinate) - coordinate
    else:
        return coordinate
    
# Symetrically extended borders
def symetricallyExtendedBordersKernel(img:np.ndarray, neigborhood_coordinates:list, kernel:np.ndarray):
    xmax = img.shape[0] - 1
    ymax = img.shape[1] - 1
    neigborhood_values = np.ones((kernel.shape[0], kernel.shape[1]), dtype=np.int16)
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            x = symetricalcoordinate(neigborhood_coordinates[i][j][0], xmax)
            y = symetricalcoordinate(neigborhood_coordinates[i][j][1], ymax)
            neigborhood_values[i, j] = img[x, y]
    return int(np.sum(neigborhood_values * kernel))

####Filters####
# average image filter
def average_filter(img: np.ndarray, kernel_size:int):
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
def gaussian_filter(img:np.ndarray, sigma:float, kernel_size:int):
    gaussianKernel = gaussian_kernel(kernel_size, sigma)
    
    seb = np.copy(img)
    peb = np.copy(img)
    zp = np.copy(img)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            neigborhood_coordinates = neighborhoodCoordinates(i, j, kernel_size, sorted=True)
            seb[i,j] = symetricallyExtendedBordersKernel(img, neigborhood_coordinates, gaussianKernel)
            peb[i,j]  = periodicallyExtendedBordersKernel(img, neigborhood_coordinates, gaussianKernel)
            zp[i,j]   = zeroPaddingKernel(img, neigborhood_coordinates, gaussianKernel)
            
    # Show Images Result
    fig = plt.figure(figsize=(15, 5))
    plt.title('Gaussian Filter')
    plt.axis('off')
    fig.add_subplot(1, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title('Original Image')
    
    fig.add_subplot(1, 4, 2)
    plt.imshow(seb, cmap='gray')
    plt.axis('off')
    plt.title('Symetrically Extended Borders')
    
    fig.add_subplot(1, 4, 3)
    plt.imshow(peb, cmap='gray')
    plt.axis('off')
    plt.title('Periodically Extended Borders')
    
    fig.add_subplot(1, 4, 4)
    plt.imshow(zp, cmap='gray')
    plt.axis('off')
    plt.title('Zero Padding')
    
    fig.show()
    
# Laplacian filter
def laplace_filter(img:np.ndarray, kernel:np.ndarray):
    rb = np.copy(img)
    peb = np.copy(img)
    zp = np.copy(img)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            neigborhood_coordinates = neighborhoodCoordinates(i, j, kernel.shape[0], sorted=True)
            rb[i,j] = replicatedBordersKernel(img, neigborhood_coordinates, kernel)
            peb[i,j]  = periodicallyExtendedBordersKernel(img, neigborhood_coordinates, kernel)
            zp[i,j]   = zeroPaddingKernel(img, neigborhood_coordinates, kernel)
    
    # Show Images Result
    fig = plt.figure(figsize=(15, 5))
    plt.title('Laplace Filter')
    plt.axis('off')
    fig.add_subplot(1, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title('Original Image')
    
    fig.add_subplot(1, 4, 2)
    plt.imshow(rb, cmap='gray')
    plt.axis('off')
    plt.title('Replicated Borders')
    
    fig.add_subplot(1, 4, 3)
    plt.imshow(peb, cmap='gray')
    plt.axis('off')
    plt.title('Periodically Extended Borders')
    
    fig.add_subplot(1, 4, 4)
    plt.imshow(zp, cmap='gray')
    plt.axis('off')
    plt.title('Zero Padding')
    
    fig.show()
    
if __name__ == "__main__":
    image_path = Path("img/68.png")
    img = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
    average_filter(img, 3)
    gaussian_filter(img, 0.5, 3)
    laplace_filter(img, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]))
    laplace_filter(img, np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]]))
    input("Press Enter to continue...")