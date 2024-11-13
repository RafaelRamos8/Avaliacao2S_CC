import cv2
import numpy as np

#carregar a imagem
imagem = cv2.imread('coffe.jpeg', cv2.IMREAD_GRAYSCALE)

#Aplicar o filtro gaussiano com o tamanho de kernel 25x25
filtro_gaussiano = cv2.GaussianBlur(imagem, (25, 25), 0)

#Mostrar a imagem original e a imagem filtrada
cv2.imshow('Imagem Original', imagem)
cv2.imshow('Imagem com filtro Gaussiano', filtro_gaussiano)

#esperar até que qualquer tecla seja pressionada e fechar as janelas
cv2.waitKey(0)
cv2.destroyAllWindows()