import cv2
import numpy as np
import pywt

# Função para aplicar o filtro de mediana
def filtro_mediana(img, ksize=5):
    img_mediana = cv2.medianBlur(img, ksize)
    return img_mediana

# Função para aplicar a Transformada de Haar e retornar a aproximação LL
def transformada_haar(img):
    coeffs2 = pywt.dwt2(img, 'haar')  # Decomposição da imagem
    LL, (LH, HL, HH) = coeffs2  # LL é a aproximação, LH, HL e HH são os detalhes
    # Normaliza para a faixa [0, 255] e converte para uint8
    LL = np.uint8(255 * (LL - np.min(LL)) / np.ptp(LL))
    return LL

# Função para aplicar filtro bilateral
def filtro_bilateral(img):
    # Aplica filtro bilateral
    return cv2.bilateralFilter(img, 9, 25, 25)

# Função de aguçamento (filtro de alta frequência)
def filtro_agucamento(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Filtro de aguçamento
    return cv2.filter2D(img, -1, kernel)

# Função para aumentar a nitidez utilizando os filtros existentes
def aumentar_nitidez_com_filtros(img):
    # Aplica o filtro de mediana para suavizar a imagem e remover ruídos
    img_mediana = filtro_mediana(img)

    # Aplica a decomposição wavelet (Haar) para capturar as características de alta frequência
    LL = transformada_haar(img_mediana)

    # Aplica o filtro de aguçamento na imagem resultante da decomposição Haar
    img_agucada = filtro_agucamento(LL)

    # Retorna a imagem com aumento de nitidez
    return img_agucada

# Carrega a imagem em escala de cinza
img = cv2.imread('imagem.jpeg', cv2.IMREAD_GRAYSCALE)

# Aplica os filtros sequenciais para aumentar a nitidez
img_nitida = aumentar_nitidez_com_filtros(img)

# Redimensiona a imagem de nitidez para o tamanho original
img_nitida_resized = cv2.resize(img_nitida, (img.shape[1], img.shape[0]))

# Combina a imagem original e a imagem melhorada para visualização lado a lado
img_resized = cv2.resize(img, None, fx=0.5, fy=0.5)
img_nitida_resized = cv2.resize(img_nitida_resized, None, fx=0.5, fy=0.5)

# Exibe as imagens original e melhorada lado a lado
imagens_combinadas = np.hstack((img_resized, img_nitida_resized))

# Exibe a imagem combinada
cv2.imshow("Imagem Original e Melhorada", imagens_combinadas)

# Aguarda a tecla para fechar
cv2.waitKey(0)
cv2.destroyAllWindows()
