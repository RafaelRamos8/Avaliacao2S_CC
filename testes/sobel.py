import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem em escala de cinza
imagem = cv2.imread('coffe.jpeg', cv2.IMREAD_GRAYSCALE)

#Aplicar o filtro de sobel na direção x (encontrar bordas verticais)
sobel_x = cv2.Sobel(imagem, cv2.CV_64F, 0,1, ksize=3)

#Aplicar o filtro de sobel na direçao y (encontrar bordas horizontais)
sobel_y = cv2.Sobel(imagem, cv2.CV_64F, 0,1, ksize=3)

#Converter os resultados para um formato mais facil de visualizar
sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)

#Combinar as bordas encontradas nas direções X e Y
sobel_combinado = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

#Mostrar os resultados usando Matplotlib
plt.figure(figsize=(10, 6))

plt.subplot(1, 3, 1)
plt.title('Imagem Original')
plt.imshow(imagem, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Filtro Sobel X')
plt.imshow(sobel_x, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Filtro Sobel Y')
plt.imshow(sobel_y, cmap='gray')
plt.axis('off')

plt.figure()
plt.title('Sobel Combinado')
plt.imshow(sobel_combinado, cmap='gray')
plt.axis('off')

plt.show()