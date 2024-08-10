import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


@st.cache_data

def load_data():
    df_hi = pd.read_csv('heroes_information.csv', index_col=0)
    df_spr = pd.read_csv('super_hero_powers.csv')
    return df_hi, df_spr


st.header("Aplicação - Dados de Super-Heróis")
df1, df2 = load_data()

st.sidebar.write("Exibir")

# Checkbox - Exibe Dados
checkbox_data = st.sidebar.checkbox("Dados")

if checkbox_data:
    st.header("Dados:")
    st.header("Dataset `heroes_information.csv`")
    st.data_editor(df1) # Exibe o dataframe todo
    
    # Exibe partes selecionadas
    super_hero_df1 = st.multiselect('Selecione o(s) Super-Herói(s):', list(df1['name']))
    df1_super_hero = df1[df1['name'].isin(super_hero_df1)]
    columns_df1 = st.multiselect('Selecione a(s) coluna(s):', df1.drop(columns=['name']).columns) # A coluna 'name' já será exibida
    df1_filtered = df1_super_hero[columns_df1]  
    if super_hero_df1 and columns_df1:
        df1_filtered.insert(0, 'name', df1_super_hero['name'])
        st.data_editor(df1_filtered)
    
    st.header("Dataset `super_hero_powers.csv`")
    st.data_editor(df2) # Exibe o dataframe todo
    
    # Exibe partes selecionadas
    super_hero_df2 = st.multiselect('Selecione o(s) Super-Herói(s):', list(df2['hero_names']))
    df2_super_hero = df2[df2['hero_names'].isin(super_hero_df2)]
    columns_df2 = st.multiselect('Selecione a(s) coluna(s):', df2.drop(columns=['hero_names']).columns) # A coluna 'name' já será exibida
    df2_filtered = df2_super_hero[columns_df2]
    if super_hero_df2 and columns_df2:
        df2_filtered.insert(0, 'hero_names', df2_super_hero['hero_names'])
        st.data_editor(df2_filtered)
    

# Checkbox - Exibe Estatísticas
checkbox_stats = st.sidebar.checkbox("Estatísticas")

if checkbox_stats: 
    st.header("Estatísticas Descritivas:")
    st.header("Dataset `heroes_information.csv`")
    st.write(df1.describe())
    st.header("Dataset `super_hero_powers.csv`")
    st.write(df2.describe())   

# Checkbox - Exibe Distribuições
checkbox_distr = st.sidebar.checkbox("Distribuições")


# Função para criar o histograma
def plot_histogram_num(data, column_name):
    plt.figure(figsize=(10, 6))        
    plt.hist(data[column_name], bins=30, edgecolor='black')
    plt.title(f'Histograma da Coluna {column_name}')
    plt.xlabel('Valor')
    plt.ylabel('Frequência')
    st.pyplot(plt)


def plot_histogram_other(counts_in, column_name):
    fig, ax = plt.subplots()
    counts_in.plot(kind='bar', ax=ax, color=['blue', 'orange'])

    # Adiciona título e rótulos aos eixos
    ax.set_title(f'Histograma da coluna {column_name}')
    ax.set_xlabel('Valor')
    ax.set_ylabel('Frequência')

    # Mostra o gráfico no Streamlit
    st.pyplot(fig)
    
if checkbox_distr:
    st.header("Distribuições:")
    st.header("Dataset `heroes_information.csv`")
    column_name1 = st.selectbox('Selecione a coluna para exibir o histograma:', 
                               ['Gender', 'Eye color', 'Race', 'Hair color', 'Height', 'Publisher', 'Skin color', 'Alignment', 'Weight'])
    if column_name1 in ['Height', 'Weight']:
        plot_histogram_num(df1, column_name1)
    else:
        counts = df1[column_name1].value_counts()
        plot_histogram_other(counts, column_name1)
    
    st.header("Dataset `super_hero_powers.csv`")
    df2_temp = df2.copy()
    df2_temp.drop(['hero_names'], axis=1, inplace=True)
    
    column_name2 = st.selectbox('Selecione a coluna para exibir o histograma:', 
                               df2_temp.columns)
    counts2 = df2_temp[column_name2].value_counts()
    plot_histogram_other(counts2, column_name2)


# Checkbox - Exibe Filtragens
checkbox_filter = st.sidebar.checkbox("Filtragens")

if checkbox_filter:
    
    st.header("Filtragens:")
    st.header("Dataset `heroes_information.csv`")
    column_name3 = st.selectbox('Selecione a coluna para fazer a filtragem:', 
                               ['Gender', 'Publisher', 'Alignment'])
    
    # Filtragem baseada na coluna
    unique_values = df1[column_name3].unique()
    unique_selected = st.multiselect('Selecione pelo menos uma informação para a filtragem:', unique_values)

    # Filtrar o DataFrame
    df_filtered = df1[df1[column_name3].isin(unique_selected)]

    # Exibe o DataFrame filtrado
    if unique_selected:
        st.dataframe(df_filtered)

# Checkbox - Exibe Agrupamentos
checkbox_cluster = st.sidebar.checkbox("Resultados dos Agrupamentos")

if checkbox_cluster:
    df_cluster = pd.read_csv('df_clustering.csv')

    st.header("Resultado dos Agrupamentos:")
    cluster_selected = st.multiselect('Selecione o Cluster para Visualizar o Conjunto de Dados:', 
                                      [0, 1])
    
    # Filtra o DataFrame
    df_cluster_filtered = df_cluster[df_cluster['cluster'].isin(cluster_selected)]

    # Exibe o DataFrame filtrado
    if cluster_selected:
        st.dataframe(df_cluster_filtered)

    # Mapa de Calor
    st.header("Mapa de Calor:")

    # Filtrado pelo cluster escolhido
    cluster_selected2 = st.selectbox('Selecione o Cluster:', [0, 1])
    df_cluster_filtered2 = df_cluster[df_cluster['cluster'] == cluster_selected2]

    # Convertendo valores booleanos para inteiros (0 e 1)
    df_bool = df_cluster_filtered2.drop(['hero_names', 'cluster'], axis=1).astype(int)  # Todas as colunas exceto 'hero_names' e 'cluster'

    # Escolhendo as features para o mapa de calor:
    feats = st.multiselect('Selecione as features:', 
                            df_bool.columns)
    
    if feats:
        # Calculando a correlação entre as features booleanas
        df_bool2 = df_bool[feats]

        correlation_matrix = df_bool2.corr()

        # Gerando o mapa de calor
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', cbar=True, square=True)
        plt.title(f'Mapa de Calor das Features - Cluster {cluster_selected2}')

        # Mostrar o mapa de calor no Streamlit
        st.pyplot(plt)


# Checkbox - Exibe Classificações
checkbox_classification = st.sidebar.checkbox("Resultado das Classificações")

if checkbox_classification:
    st.header("Resultado das Classificações")
    
    # Carrega os dados tratados no Jupyter Notebook
    df_classification = pd.read_csv('df_encoded_classification.csv')
    
    # Decodifica o rótulo
    label_mapping = {0: 'bad', 1: 'good'}
    df_classification['Alignment'] = df_classification['Alignment'].replace(label_mapping)
    
    # Carrega o modelo salvo
    classification_model = joblib.load('model_classification.pkl')
    
    # Seleciona o super-herói e filtra o dataset:
    super_hero_class = st.selectbox('Selecione o Super-Herói:', list(df_classification['name']))
    df_super_hero_class = df_classification[df_classification['name'] == super_hero_class]

    # Prevê a classe do super-herói pelo modelo treinado
    result_classif = classification_model.predict(df_super_hero_class.drop(columns=['name', 'Eye color', 'Hair color', 'Publisher', 'Skin color', 'Alignment']))  
    st.write(f'Classificação predita: {label_mapping[result_classif.item()]}')
    st.write(f"Classificação real: {df_super_hero_class['Alignment'].item()}")
    
    if label_mapping[result_classif.item()] == df_super_hero_class['Alignment'].item():
        st.write('✅ : Modelo acertou')
    else:
        st.write('❌ : Modelo errou')

# Checkbox - Exibe Previsões Peso
checkbox_regression = st.sidebar.checkbox("Resultado das Regressões")

if checkbox_regression:
    
    st.header("Resultado das Regressões")
    
    df_reg_original = pd.read_csv('df_encoded_regression.csv')
    
    # Remove a coluna Weight e insere as dummy para Alignment
    df_reg = df_reg_original.drop(columns=['Weight'])
    
    # Carrega o modelo salvo
    regression_model = joblib.load('model_regression.pkl')
    
    # Seleciona o super-herói e filtra o dataset:
    super_hero_reg = st.selectbox('Selecione o Super-Herói:', list(df_reg['name']))
    df_super_hero_reg = df_reg[df_reg['name'] == super_hero_reg]
    
    # st.data_editor(df_reg_original)

    # Prevê o peso do super-herói pelo modelo treinado
    result_reg = regression_model.predict(df_super_hero_reg.drop(columns=['name', 'Eye color', 'Hair color', 'Publisher', 'Skin color']))  
        
    # st.write(result_reg)
    
    result_reg_adapt = np.array([result_reg[0], result_reg[0]])
    
    # Carrega e Decodifica com o StandardScaler do valor do peso:
    scaler = joblib.load('scaler.pkl')       
    result_reg_decod = scaler.inverse_transform(result_reg_adapt.reshape(1, -1))

    df_reg_original_super_hero = df_reg_original[df_reg_original['name'] == super_hero_reg] # Filtrando o df para aquele super-herói
    result_decod = scaler.inverse_transform(df_reg_original_super_hero[['Weight', 'Height']])
    result_decod_df = pd.DataFrame(result_decod, columns=['Weight', 'Height']) # Convertendo o array para um DataFrame
    
    real_weight = result_decod_df['Weight'].item()
    predicted_weight = result_reg_decod[0, 0]
    
    st.write(f"Peso predito: {predicted_weight:.2f}")
    st.write(f"Peso real: {real_weight}")
    st.write(f"Diferença de {abs(predicted_weight - real_weight):.2f}")