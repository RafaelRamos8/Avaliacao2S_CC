import numpy as np
import cv2
import pywt


# Função para aplicar o filtro Laplaciano
def laplaciano(img):
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    return cv2.filter2D(img, -1, kernel)


# Função para aplicar o filtro Gaussiano
def gaussiano(img, tamanho_kernel=5, sigma=1):
    return cv2.GaussianBlur(img, (tamanho_kernel, tamanho_kernel), sigma)


# Função para aplicar o filtro de média
def media(img, tamanho_kernel=5):
    return cv2.blur(img, (tamanho_kernel, tamanho_kernel))


# Função para aplicar o filtro de mediana
def mediana(img, tamanho_kernel=5):
    return cv2.medianBlur(img, tamanho_kernel)


# Função para aplicar o filtro Sobel
def sobel(img):
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return cv2.magnitude(grad_x, grad_y)


# Função para aumentar a acuidade da imagem
def agucamento(img):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)


# Função otimizada para aplicar o ruído sal e pimenta
def ruido_sal_pimenta(img, probabilidade=0.02):
    output = img.copy()

    # Criar máscara de ruído
    rnd = np.random.random(img.shape)

    # Aplicar sal (pixels brancos)
    output[rnd < probabilidade / 2] = 255

    # Aplicar pimenta (pixels pretos)
    output[rnd > 1 - probabilidade / 2] = 0

    return output


# Função melhorada para aplicar ruído gaussiano
def ruido_gaussiano(img, media=0, sigma=25):
    # Garantir que a imagem está em float64 para evitar overflow
    img_float = img.astype(np.float64)

    # Gerar ruído gaussiano
    gauss = np.random.normal(media, sigma, img.shape)

    # Adicionar ruído à imagem
    noisy = img_float + gauss

    # Normalizar valores para o intervalo [0, 255]
    noisy = np.clip(noisy, 0, 255)

    return noisy.astype(np.uint8)


# Função para calcular a transformada de Fourier com filtro passa-baixa
def transformada_fourier(img):
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    # Transformada de Fourier e centralização do espectro
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    # Máscara de filtro passa-baixa (experimente ajustar o valor "50")
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow - 50:crow + 50, ccol - 50:ccol + 50] = 1

    # Aplicar filtro e realizar a transformada inversa
    fshift_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.abs(np.fft.ifft2(f_ishift))

    # Retornar o espectro de magnitude em escala logarítmica para visualização
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    return img_back, magnitude_spectrum


# Função para aplicar a transformada de Haar usando PyWavelets
def transformada_haar(img):
    coeffs2 = pywt.dwt2(img, 'haar')  # Decomposição da imagem
    LL, (LH, HL, HH) = coeffs2  # LL é a aproximação, LH, HL e HH são os detalhes
    return LL, LH, HL, HH