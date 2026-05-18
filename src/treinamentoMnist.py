import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers
from tensorflow.keras.datasets import mnist

# ─────────────────────────────────────────
# DADOS — pipeline com tf.data
# ─────────────────────────────────────────
# Carrega o dataset MNIST dividido em dados de treino (60.000 imagens) e teste (10.000 imagens)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalização: Transforma os pixels (que vão de 0 a 255) em valores entre 0.0 e 1.0.
# O '[..., np.newaxis]' adiciona a dimensão de "canal de cor" (1 para escala de cinza), exigido pelo Keras. 
X_train = X_train[..., np.newaxis].astype("float32") / 255.0
X_test  = X_test [..., np.newaxis].astype("float32") / 255.0

## One-hot encoding: Transforma as saídas (ex: número 3) em um vetor de 10 posições: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
Y_train = keras.utils.to_categorical(y_train, 10)
Y_test  = keras.utils.to_categorical(y_test,  10)

# Pipeline otimizado com tf.data: Configura como os dados são entregues à memória do computador.
BATCH_SIZE = 64 # O modelo vai olhar 64 imagens por vez antes de atualizar seus pesos
AUTOTUNE   = tf.data.AUTOTUNE # Deixa o TensorFlow gerenciar o uso de CPU/GPU automaticamente

# train_ds: Mistura as imagens (shuffle), divide em lotes (batch) e prepara o próximo lote em segundo plano (prefetch)
train_ds = (tf.data.Dataset.from_tensor_slices((X_train, Y_train))
            .shuffle(10000)
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE))

test_ds = (tf.data.Dataset.from_tensor_slices((X_test, Y_test))
           .batch(BATCH_SIZE)
           .prefetch(AUTOTUNE))

# ─────────────────────────────────────────
# DATA AUGMENTATION
# ─────────────────────────────────────────
# Cria uma mini-rede que rotaciona, aproxima (zoom) e move as imagens levemente.
# Isso ajuda o modelo a não "decorar" as imagens de treino e se tornar mais robusto.

data_augmentation = keras.Sequential([
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
], name="augmentation")

# ─────────────────────────────────────────
# MODELO — API Funcional + BatchNorm
# ─────────────────────────────────────────
def meu_modelo():
    # Define o formato de entrada: imagens de 28x28 pixels com 1 canal de cor
    inputs = keras.Input(shape=(28, 28, 1), name="input")

    # Aplica as distorções visuais do Data Augmentation apenas na fase de treino
    x = data_augmentation(inputs)

    # Bloco 1: Procura padrões simples (linhas, bordas) na imagem
    x = layers.Conv2D(32, (3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, (3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    # Bloco 2: Procura padrões mais complexos (curvas, junções)
    x = layers.Conv2D(64, (3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, (3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    # Bloco 3 : Extrai as características finais da imagem 
    x = layers.Conv2D(128, (3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)   # substitui Flatten + Dense pesado
    x = layers.Dropout(0.5)(x)

    # Camada de Saída: Dá a resposta final com as probabilidades para cada um dos 10 números (0 a 9)
    # A ativação "softmax" garante que a soma de todas as probabilidades seja 100%
    outputs = layers.Dense(10, activation="softmax", name="output")(x)

    return keras.Model(inputs, outputs, name="MNIST_Moderno")

modelo = meu_modelo()
modelo.summary()

# ─────────────────────────────────────────
# 4. CALLBACKS
# ─────────────────────────────────────────
cb_list = [
    # Salva o melhor modelo em formato moderno
    callbacks.ModelCheckpoint(
        "modeloTOP.keras",          # .keras substituiu .h5
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    # Para o treino se não melhorar
    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    # Reduz o learning rate quando estagna
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
]

# ─────────────────────────────────────────
# 5. COMPILAÇÃO com API atual
# ─────────────────────────────────────────
modelo.compile(
    loss="categorical_crossentropy",
    optimizer=optimizers.Adam(learning_rate=1e-3),  # lr depreciado
    metrics=["accuracy"]
)

# ─────────────────────────────────────────
# 6. TREINAMENTO
# ─────────────────────────────────────────
history = modelo.fit(
    train_ds,
    epochs=50,              # EarlyStopping vai parar antes se necessário
    validation_data=test_ds,
    callbacks=cb_list,
)

# ─────────────────────────────────────────
# 7. AVALIAÇÃO
# ─────────────────────────────────────────
loss, acc = modelo.evaluate(test_ds, verbose=0)
print(f"Test loss:     {loss:.4f}")
print(f"Test accuracy: {acc*100:.2f}%")

# ─────────────────────────────────────────
# 8. VISUALIZAÇÃO do treinamento
# ─────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history["accuracy"],     label="Treino")
ax1.plot(history.history["val_accuracy"], label="Validação")
ax1.set_title("Acurácia")
ax1.set_xlabel("Época")
ax1.legend()

ax2.plot(history.history["loss"],     label="Treino")
ax2.plot(history.history["val_loss"], label="Validação")
ax2.set_title("Loss")
ax2.set_xlabel("Época")
ax2.legend()

plt.tight_layout()
plt.show()

# ─────────────────────────────────────────
# 9. PREDIÇÃO com visualização
# ─────────────────────────────────────────
idx = 50
img = X_test[idx]
pred = modelo.predict(img[np.newaxis, ...], verbose=0)
classe = np.argmax(pred)
confianca = pred[0][classe] * 100

plt.imshow(img.squeeze(), cmap="gray")
plt.title(f"Previsto: {classe}  |  Real: {np.argmax(Y_test[idx])}  |  Confiança: {confianca:.1f}%")
plt.axis("off")
plt.show()