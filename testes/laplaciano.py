import cv2
import numpy as np
import matplotlib.pyplot as plt

#carrega a imagem em tons de cinza
imagem = cv2.imread('cox.png', cv2.IMREAD_GRAYSCALE)

#Aplica o filtro laplaciano
laplaciano = cv2.Laplacian(imagem, cv2.CV_64F)

#converte a imagem filtrada para o tipo uint8 (valores entre 0 e 255)
laplaciano = cv2.convertScaleAbs(laplaciano)

#mostrar a imagem original e a imagem com o filtro aplicado lado a lado
plt.figure(figsize=(10,5))

plt.subplot(1, 2, 1)
plt.title('Imagem Original')
plt.imshow(imagem, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Filtro Laplaciano')
plt.imshow(laplaciano, cmap = 'gray')
plt.axis('off')

plt.tight_layout()
plt.show()