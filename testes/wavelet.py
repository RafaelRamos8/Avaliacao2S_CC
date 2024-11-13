import cv2
import pywt
import matplotlib.pyplot as plt
# carregar um imagem
imagem = cv2.imread('raio.jpeg', cv2.IMREAD_GRAYSCALE)
# realizae a decomposição da imagem em waveltes
coeffs2 = pywt.dwt2(imagem, 'haar')
# usando wavelets 'haar'
LL, (LH, HL, HH) = coeffs2
# LL: aproximações - LH, HL, HH: detalhes
# mostrar resultados
plt.figure(figsize = (10,10))

plt.subplot(2, 2, 1)
plt.imshow(imagem, cmap = 'gray')
plt.title('Imagem Original')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(LL, cmap = 'gray')
plt.title('Aproximações')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(LH, cmap = 'gray')
plt.title('Detalhes (LH)')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(HL, cmap = 'gray')
plt.title('Detalhes (HL)')
plt.axis('off')

plt.tight_layout()
plt.show()