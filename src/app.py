import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
from tensorflow import keras

# CARREGAMENTO DO MODELO
# Carregamos o modelo que treinamos anteriormente
modelo = keras.models.load_model("../modeloTOP.keras")

class AppClassificador:
    def __init__(self, root):
        self.root = root
        self.root.title("Classificador de Números - Parceiro de Programação")
        
        # Criação de uma tela (Canvas) para desenhar
        self.canvas = tk.Canvas(root, width=280, height=280, bg="black", cursor="cross")
        self.canvas.grid(row=0, column=0, pady=10, padx=10, columnspan=2)
        
        # Criação de uma imagem interna no Python para guardar o desenho
        self.image = Image.new("L", (280, 280), 0)
        self.draw = ImageDraw.Draw(self.image)
        
        # Configuração do mouse
        self.canvas.bind("<B1-Motion>", self.desenhar)
        
        # Botões
        self.btn_prever = tk.Button(root, text="Adivinhar Número", command=self.classificar)
        self.btn_prever.grid(row=1, column=0, pady=5)
        
        self.btn_limpar = tk.Button(root, text="Limpar Tela", command=self.limpar)
        self.btn_limpar.grid(row=1, column=1, pady=5)
        
        self.label_res = tk.Label(root, text="Desenhe um número!", font=("Helvetica", 16))
        self.label_res.grid(row=2, column=0, columnspan=2, pady=10)

    def desenhar(self, event):
        # Desenha na tela e na imagem interna
        x, y = event.x, event.y
        r = 8 # Espessura do traço
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=255)

    def limpar(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.label_res.config(text="Desenhe um número!")

    def classificar(self):
        # Redimensiona para 28x28 (tamanho que o modelo MNIST espera)
        img_reduzida = self.image.resize((28, 28))
        
        # Converter para array do NumPy e normalizar (0 a 1)
        img_array = np.array(img_reduzida).astype("float32") / 255.0
        
        # Ajustar o formato para (1, 28, 28, 1) - Batch, Altura, Largura, Canal
        img_final = img_array[np.newaxis, ..., np.newaxis]
        
        # Fazer a previsão
        pred = modelo.predict(img_final, verbose=0)
        resultado = np.argmax(pred)
        confianca = np.max(pred) * 100
        
        self.label_res.config(text=f"Previsão: {resultado} ({confianca:.1f}%)")

# Iniciar a aplicação
if __name__ == "__main__":
    root = tk.Tk()
    app = AppClassificador(root)
    root.mainloop()