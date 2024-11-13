import cv2
import numpy as np

# Função para aplicar o filtro de mediana
def filtro_mediana(img, ksize=3):
    img_mediana = cv2.medianBlur(img, ksize)
    return img_mediana

# Função para aplicar a Transformada de Fourier e retornar à imagem espacial
def transformada_fourier(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    img_back = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(img_back)
    img_back = np.abs(img_back)
    
    # Normaliza a imagem para o intervalo [0, 255] e converte para uint8
    img_back = np.uint8(255 * img_back / np.max(img_back))
    return img_back

# Função para aplicar o filtro de aguçamento
def filtro_agucamento(img):
    kernel = np.array([[0, -1, 0], [-1.03, 5, -1.03], [0, -1, 0]])
    img_agucado = cv2.filter2D(img, -1, kernel)
    return img_agucado

# Função para aplicar o filtro bilateral
def filtro_bilateral(img):
    # Aplica filtro bilateral para suavizar a imagem, preservando as bordas
    return cv2.bilateralFilter(img, 5, 15, 15)

# Carrega a imagem em escala de cinza
img = cv2.imread('imagem.jpeg', cv2.IMREAD_GRAYSCALE)

# Aplica os filtros sequencialmente na mesma imagem
img_mediana = filtro_mediana(img)
img_bilateral = filtro_bilateral(img_mediana)  # Filtro bilateral após mediana
img_fourier = transformada_fourier(img_bilateral)  # Aplica a Transformada de Fourier
img_agucamento = filtro_agucamento(img_fourier)  # Aplica o filtro de aguçamento

# Redimensiona as imagens para 50% do tamanho original
img_resized = cv2.resize(img_agucamento, None, fx=0.5, fy=0.5)
img_original_resized = cv2.resize(img, None, fx=0.5, fy=0.5)

# Empilha as imagens (original e final) lado a lado
imagem_combinada = np.hstack((img_original_resized, img_resized))

# Exibe a imagem combinada
cv2.imshow("Antes e Depois", imagem_combinada)

# Aguarda a tecla para fechar
cv2.waitKey(0)
cv2.destroyAllWindows()
