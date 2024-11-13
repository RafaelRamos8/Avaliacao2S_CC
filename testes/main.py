import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ruidos_e_filtros import (
    ruido_sal_pimenta, ruido_gaussiano, laplaciano, gaussiano,
    media, mediana, sobel, agucamento, transformada_fourier, transformada_haar
)


# Função para exibir as combinações de filtros e ruídos
def exibir_combinacoes(img):
    # Aplica os filtros e gera as imagens processadas
    img_laplaciano = laplaciano(img)
    img_gaussiano_blur = gaussiano(img)
    img_media = media(img)
    img_mediana = mediana(img)
    img_sobel = sobel(img)
    img_agucamento = agucamento(img)

    # Aplica os ruídos
    img_sal_pimenta = ruido_sal_pimenta(img.copy())
    img_gaussiano = ruido_gaussiano(img.copy())
    img_combined_noise = ruido_sal_pimenta(ruido_gaussiano(img.copy()))

    # Aplica a transformada de Fourier
    img_back_fourier, magnitude_fourier = transformada_fourier(img)
    img_back_fourier_sp = transformada_fourier(img_sal_pimenta)  # Fourier + Sal e Pimenta
    img_back_fourier_gauss = transformada_fourier(img_gaussiano)  # Fourier + Gaussiano

    # Aplica a transformada Haar
    LL_wavelet, LH_wavelet, HL_wavelet, HH_wavelet = transformada_haar(img)
    LL_wavelet_sp, LH_wavelet_sp, HL_wavelet_sp, HH_wavelet_sp = transformada_haar(
        img_sal_pimenta)  # Haar + Sal e Pimenta
    LL_wavelet_gauss, LH_wavelet_gauss, HL_wavelet_gauss, HH_wavelet_gauss = transformada_haar(
        img_gaussiano)  # Haar + Gaussiano

    # Inicia a janela principal do Tkinter
    root = tk.Tk()
    root.title("Exibição de Filtros e Ruídos")

    # Definindo a janela para tela cheia (opcional)
    root.state('zoomed')  # Isso maximiza a janela automaticamente

    # Cria uma barra de rolagem
    canvas = tk.Canvas(root)
    scroll_y = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scroll_y.set)

    # Cria o frame onde as imagens serão exibidas
    frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=frame, anchor="nw")

    # Função para ajustar o tamanho das imagens para o mesmo tamanho
    def ajustar_tamanho(imagem, tamanho=(400, 400)):
        img_pil = Image.fromarray(np.uint8(imagem))
        img_pil = img_pil.resize(tamanho, Image.Resampling.LANCZOS)  # Atualizado para usar LANCZOS
        return ImageTk.PhotoImage(img_pil)

    # Adiciona as imagens no frame
    imagens = [
        (img, 'Imagem Original'),
        (img_laplaciano, 'Laplaciano'),
        (img_gaussiano_blur, 'Gaussiano'),
        (img_media, 'Filtro Média'),
        (img_mediana, 'Filtro Mediana'),
        (img_sobel, 'Filtro Sobel'),
        (img_agucamento, 'Filtro Aguçamento'),
        (img_sal_pimenta, 'Sal e Pimenta'),
        (img_gaussiano, 'Ruído Gaussiano'),
        (img_combined_noise, 'Sal e Pimenta + Gaussiano'),
        (magnitude_fourier, 'Magnitude Fourier'),
        (img_back_fourier, 'Fourier Processada'),
        (img_back_fourier_sp, 'Fourier + Sal e Pimenta'),
        (img_back_fourier_gauss, 'Fourier + Gaussiano'),
        (LL_wavelet, 'Wavelet LL'),
        (LH_wavelet, 'Wavelet LH'),
        (HL_wavelet, 'Wavelet HL'),
        (HH_wavelet, 'Wavelet HH'),
        (LL_wavelet_sp, 'Wavelet LL + Sal e Pimenta'),
        (LH_wavelet_sp, 'Wavelet LH + Sal e Pimenta'),
        (HL_wavelet_sp, 'Wavelet HL + Sal e Pimenta'),
        (HH_wavelet_sp, 'Wavelet HH + Sal e Pimenta'),
        (LL_wavelet_gauss, 'Wavelet LL + Gaussiano'),
        (LH_wavelet_gauss, 'Wavelet LH + Gaussiano'),
        (HL_wavelet_gauss, 'Wavelet HL + Gaussiano'),
        (HH_wavelet_gauss, 'Wavelet HH + Gaussiano'),
    ]

    # Função para adicionar as imagens ao frame
    def mostrar_imagens(imagens):
        row_idx = 0  # Inicializa o índice de linha para o grid
        col_idx = 0  # Inicializa o índice de coluna para o grid

        for i, (img_item, titulo) in enumerate(imagens):
            # Verifique se a imagem é válida (np.ndarray e com tipo correto)
            if isinstance(img_item, np.ndarray):
                if img_item.ndim == 2:  # Imagem em escala de cinza
                    img_tk = ajustar_tamanho(img_item)
                elif img_item.ndim == 3:  # Imagem colorida
                    img_tk = ajustar_tamanho(img_item)
                else:
                    print(f"Formato inválido para a imagem: {titulo}")
                    continue
            else:
                print(f"Tipo inválido para a imagem: {titulo}")
                continue

            # Cria o container para cada imagem e título
            title_label = tk.Label(frame, text=titulo, font=("Arial", 12), anchor="n", width=20, height=2)
            title_label.grid(row=row_idx, column=col_idx, padx=10, pady=5, sticky="n")

            img_label = tk.Label(frame, image=img_tk)
            img_label.image = img_tk
            img_label.grid(row=row_idx + 1, column=col_idx, padx=10, pady=10)

            # Atualiza os índices de linha e coluna
            col_idx += 1
            if col_idx == 2:  # Após duas imagens, passa para a próxima linha
                col_idx = 0
                row_idx += 2  # Incrementa 2 para deixar espaço para título e imagem

    # Mostrar as imagens
    mostrar_imagens(imagens)

    # Atualiza o tamanho do frame para se ajustar ao conteúdo
    frame.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))

    # Coloca a barra de rolagem
    canvas.grid(row=0, column=0, sticky="nsew")
    scroll_y.grid(row=0, column=1, sticky="ns")

    # Expande a tela para preencher a área
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)

    # Inicia o loop da interface gráfica
    root.mainloop()


if __name__ == "__main__":
    # Carregar a imagem
    img = cv2.imread('imagem.jpg', cv2.IMREAD_GRAYSCALE)

    # Verificar se a imagem foi carregada corretamente
    if img is None:
        print("Erro ao carregar a imagem!")
    else:
        # Exibir as combinações de filtros e ruídos
        exibir_combinacoes(img)