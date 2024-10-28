# Importações das bibliotecas necessárias
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from transformers import pipeline

# Carregue os dados
df = pd.read_csv("C:\\Users\\Administrator\\Downloads\\projetoportfolio\\portfolio\\precos.csv", sep=";")


# Modelos para diferentes categorias de viagens
modelos = {}
categorias = ["Econômico", "Conforto", "Luxo"]

# Ajuste o modelo de regressão linear para cada categoria
for categoria in categorias:
    modelo = LinearRegression()
    x = df[["distancia"]]
    y = df["economico"] 
    modelo.fit(x, y)
    modelos[categoria] = modelo

# Título e cabeçalho
st.title("Previsão de Preços de Viagens de Uber")
st.divider()

# Seleção de categoria
categoria_selecionada = st.selectbox("Escolha a categoria do veículo", categorias)

# Input da distância do usuário
distancia = st.number_input("Digite a distância da viagem (km)", min_value=1.0, step=0.1)

# Cálculo do preço previsto para a categoria e distância escolhidas
if distancia and categoria_selecionada:
    modelo = modelos[categoria_selecionada]
    preco_previsto = modelo.predict([[distancia]])[0]
    st.write(f"O valor da viagem na categoria **{categoria_selecionada}** para uma distância de {distancia} km é de **R${preco_previsto:.2f}**.")

# Configuração do Chatbot usando transformers
st.divider()
st.header("Assistente Virtual - Pergunte sobre as viagens")

# Carrega o modelo de conversa DialoGPT
chatbot_pipeline = pipeline("text-generation", model="microsoft/DialoGPT-small")

# Função para processar a conversa com o chatbot
user_input = st.text_input("Digite sua pergunta para o assistente")

if user_input:
    conversa = chatbot_pipeline(user_input)
    resposta = conversa[0]["generated_text"]
    st.write(f"**Assistente:** {resposta}")

# Tabela comparativa de preços para várias distâncias
st.divider()
st.header("Tabela Comparativa de Preços para Diferentes Distâncias")
distancias = [1, 5, 10, 15, 20]
tabela_precos = pd.DataFrame({
    "Distância (km)": distancias,
    "Econômico": [modelos["Econômico"].predict([[d]])[0] for d in distancias],
    "Conforto": [modelos["Conforto"].predict([[d]])[0] for d in distancias],
    "Luxo": [modelos["Luxo"].predict([[d]])[0] for d in distancias]
})
st.table(tabela_precos)
