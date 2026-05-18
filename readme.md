# REDESNEURAIS — Classificador de Dígitos MNIST

Este repositório contém a implementação completa de uma rede neural convolucional profunda (CNN) para o reconhecimento de dígitos manuscritos utilizando o famoso conjunto de dados **MNIST**. O projeto foi reestruturado de forma profissional, separando a lógica de treino, inferência e interface.

## 🚀 Arquivos do Projeto (`src/`)

* **`treinamentoMnist.py`:** Script responsável por carregar o dataset, aplicar *Data Augmentation*, estruturar a rede CNN e treinar o modelo utilizando callbacks modernos (`EarlyStopping` e `ReduceLROnPlateau`).
* **`inferencia.py`:** Script focado em carregar o modelo treinado (`.keras`) e realizar previsões em imagens específicas para validar o aprendizado da IA.
* **`app.py`:** Interface ou aplicação principal para executar/demonstrar o projeto de forma integrada.

---

## 📁 Estrutura de Pastas do Repositório

```text
REDESNEURAIS/
│
├── src/
│   ├── app.py                  # Aplicação/Interface principal
│   ├── inferencia.py           # Script para testar predições isoladas
│   └── treinamentoMnist.py     # Script com a arquitetura e treino da CNN
│
├── .gitignore                  # Impede arquivos pesados e temporários no Git
├── modeloTOP.keras             # Arquivo gerado após o treino (ignorado pelo Git)
└── requirements.txt            # Dependências de bibliotecas do projeto
'''
___
## Clonar e Acessar o Projeto
Navegue até o diretório onde o seu repositório local está configurado:

cd REDESNEURAIS

## Criar e Ativar o Ambiente Virtual

Utilize o ambiente virtual para garantir que as bibliotecas não entrem em conflito com o seu sistema:

python -m venv .venv
.venv\Scripts\activate

## Instalar Dependências Necessárias

Com o seu ambiente .venv ativo no terminal, instale os pacotes:

pip install --upgrade pip
pip install -r requirements.txt

## Como Executar os Scripts

Para treinar a Inteligência Artificial do zero:

python src/treinamentoMnist.py

# Para testar a inferência do modelo:

python src/inferencia.py

## 🛠️ Tecnologias Utilizadas

Python 3.x — Linguagem base do projeto.

TensorFlow / Keras — Criação, compilação e treinamento da rede profunda.

NumPy — Manipulação algébrica das matrizes e tensores de imagem.

Matplotlib — Renderização visual dos gráficos e inspeção de imagens de teste.
