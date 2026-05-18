import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# CARREGAR O MODELO TREINADO
# O modeloTOP.keras contém toda a arquitetura e pesos aprendidos
try:
    modelo = keras.models.load_model("../modeloTOP.keras")
    print("✅ Modelo carregado com sucesso!")
except Exception as e:
    print(f"❌ Erro ao carregar o modelo: {e}")
    exit()

# CARREGAR DADOS DE TESTE
(_, _), (X_test, y_test) = keras.datasets.mnist.load_data()

X_test = X_test[..., np.newaxis].astype("float32") / 255.0

# REALIZA PREDIÇÕES EM MASSA
print("🧠 O modelo está a classificar as imagens...")
predicoes = modelo.predict(X_test, verbose=0)
classes_previstas = np.argmax(predicoes, axis=1)

# FUNÇÃO PARA EXIBIR RESULTADOS
def plotar_exemplos(indices, titulo, cor_texto="black"):
    plt.figure(figsize=(10, 10))
    plt.suptitle(titulo, fontsize=16, fontweight="bold")
    
    for i, idx in enumerate(indices[:16]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(X_test[idx].squeeze(), cmap="gray")
        
        real = y_test[idx]
        previsto = classes_previstas[idx]
        confianca = predicoes[idx][previsto] * 100
        
        plt.title(f"Real: {real} | Pred: {previsto}\n{confianca:.1f}%", 
                  color=cor_texto, fontsize=10)
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()

# MOSTRAR RESULTADOS ALEATÓRIOS
indices_aleatorios = np.random.choice(len(X_test), 16, replace=False)
plotar_exemplos(indices_aleatorios, "Amostra de Predições Aleatórias")

# MOSTRAR ERROS (Se existirem)
erros_idx = np.where(classes_previstas != y_test)[0]
if len(erros_idx) > 0:
    print(f"🔍 Foram encontrados {len(erros_idx)} erros em 10.000 imagens.")
    plotar_exemplos(erros_idx, "Exemplos onde o Modelo Falhou", cor_texto="red")
else:
    print("🏆 Perfeito! O modelo acertou tudo nesta amostra.")