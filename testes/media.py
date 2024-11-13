import cv2
import numpy as np
import matplotlib.pyplot as plt

#Lê a imagem em escala de cinza
imagem = cv2.imread('coffe.jpeg', cv2.IMREAD_GRAYSCALE)

#aplicando o filtro de media com tamanho 7x7
filtro_7X7 = cv2.blur(imagem, (7, 7))

#Aplicando o filtro de media com tamanho 15X15
filtro_15x15 = cv2.blur(imagem, (15, 15))

#aplicando filtro de media com tamanho 25X25
filtro_25x25 = cv2.blur(imagem, (25, 25))

#Mostrar as imagens com matplotlib
plt.figure(figsize=(15, 10))

#Imagem original
plt.subplot(2, 2, 1)
plt.imshow(imagem, cmap='gray')
plt.title("Imagem Original")
plt.axis("off")

#filtro 7x7
plt.subplot(2, 2, 2)
plt.imshow(filtro_7X7, cmap='gray')
plt.title("Filtro de media 7x7")
plt.axis("off")

#filtro 15x15
plt.subplot(2, 2, 3)
plt.imshow(filtro_15x15, cmap='gray')
plt.title("Filtro de media 15x15")
plt.axis("off")

#filtro 25x25
plt.subplot(2, 2, 4)
plt.imshow(filtro_25x25, cmap='gray')
plt.title("Filtro de media 25x25")
plt.axis("off")

plt.show()