import cmath
from math import log, ceil
import cv2
import numpy as np
import matplotlib.pyplot as plt


def omega(p, q):
    return cmath.exp((2.0 * cmath.pi * 1j * q) / p)

def zero_padding(x):
    m, n = np.shape(x)
    M, N = 2 ** int(ceil(log(m, 2))), 2 ** int(ceil(log(n, 2)))
    if (M>N):
        N=M
    else:
        M=N
    F = np.zeros((M,N), dtype = x.dtype)
    F[0:m, 0:n] = x
    return F, m, n

def fft(x):
    n = len(x)
    if n == 1:
        return x
    Feven, Fodd = fft(x[0::2]), fft(x[1::2])
    combined = [0] * n
    for m in range(n//2):
        combined[m] = Feven[m] + omega(n, -m) * Fodd[m]
        combined[m + n//2] = Feven[m] - omega(n, -m) * Fodd[m]
    return combined

def fft2(f):
    f, m, n = zero_padding(f)
    return np.transpose(fft(np.transpose(fft(f)))), m, n

def ifft2(F, m, n):
    f, M, N = fft2(np.conj(F))
    f = np.matrix(np.real(np.conj(f)))/(M*N)
    return f[0:m, 0:n]

def fftshift(F):
    M, N = F.shape
    R1, R2 = F[0: M//2, 0: N//2], F[M//2: M, 0: N//2]
    R3, R4 = F[0: M//2, N//2: N], F[M//2: M, N//2: N]
    sF = np.zeros(F.shape,dtype = F.dtype)
    sF[M//2: M, N//2: N], sF[0: M//2, 0: N//2] = R1, R4
    sF[M//2: M, 0: N//2], sF[0: M//2, N//2: N]= R3, R2
    return sF


if __name__ == "__main__":
    image = cv2.imread(r"img/81.png",0)
    plt.figure()
    plt.title("Imagen Original")
    plt.imshow(image,cmap=plt.cm.gray)
    plt.show()
    
    imagen_rellenada = zero_padding(image)[0]
    plt.figure()
    plt.title("Imagen zero-padding (Rellenada)")
    plt.imshow(imagen_rellenada, cmap=plt.cm.gray)
    plt.show()

    iT, m , n = fft2(imagen_rellenada)
    imagen_transformada = fftshift(iT)
    plt.figure()
    plt.title(f"Transformada de Fourier con relleno (Dimensiones: {imagen_transformada.shape}): ")
    plt.imshow(np.log(np.abs(imagen_transformada)), cmap=plt.cm.gray)
    plt.show()

    fr = 50
    ham = np.hamming(n)[:,None] 
    h2D = np.sqrt(np.dot(ham, ham.T)) ** fr 
    print(h2D.shape)
    plt.figure()
    plt.title("Hamming")
    plt.imshow(h2D, cmap=plt.cm.gray)
    plt.show()

    filtrada = imagen_transformada * h2D
    plt.figure()
    plt.title("Imagen filtrada: ")
    plt.imshow(np.log(np.abs(filtrada)),cmap=plt.cm.gray)
    plt.show()

    og_m, og_n = image.shape
    final_image = ifft2(filtrada, og_m, og_n)
    plt.figure()
    plt.title("Imagen final")
    plt.imshow(np.abs(final_image),cmap=plt.cm.gray)
    plt.show()