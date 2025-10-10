import streamlit as st
import pandas as pd
import plotly.express as px

# Configurar o tamanho máximo do upload para 1000 MB
st.set_page_config(
    page_title="Análise Geoestatística",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Aumentar o limite de upload
st.config.set_option('server.maxUploadSize', 1000)

# Configuração de estilo personalizado
st.markdown("""
<style>
    /* Fundo principal */
    .stApp {
        background-color: #2E2E38;
    }
    
    /* Barra lateral */
    [data-testid="stSidebar"] {
        background-color: #747480;
    }
    
    /* Botões na barra lateral */
    .stButton button {
        background-color: #FFE600 !important;
        color: black !important;
        border: none !important;
        font-weight: bold !important;
    }
    
    .stButton button:hover {
        background-color: #E6CF00 !important;
        color: black !important;
    }
    
    /* Ajuste de cores para melhor legibilidade */
    h1, h2, h3 {
        color: white !important;
    }
    
    p, label {
        color: white !important;
    }
    
    [data-testid="stMarkdownContainer"] {
        color: white;
    }
    
    /* Ajuste das cores dos selectbox */
    div[data-baseweb="select"] > div {
        background-color: #747480 !important;
        color: white !important;
    }
    
    /* Ajuste das cores dos inputs */
    .stTextInput input {
        background-color: #747480 !important;
        color: white !important;
    }
    
    /* Ajuste das bordas dos componentes */
    .stSelectbox, .stMultiSelect {
        border-color: #747480 !important;
    }
</style>
""", unsafe_allow_html=True)

def reset_data():
    st.session_state['df'] = None

if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "main"
if 'subpage' not in st.session_state:
    st.session_state['subpage'] = "univariada"

# Sidebar com navegação
st.sidebar.title("Navegação")

# Botões de navegação principal
if st.sidebar.button("Página Principal", key="main_btn", use_container_width=True):
    st.session_state['current_page'] = "main"

# Seção de Análise Exploratória
st.sidebar.markdown("### Análise Exploratória")
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Univariada", key="uni_btn", use_container_width=True):
        st.session_state['current_page'] = "geostats"
        st.session_state['subpage'] = "univariada"
with col2:
    if st.button("Multivariada", key="multi_btn", use_container_width=True):
        st.session_state['current_page'] = "geostats"
        st.session_state['subpage'] = "multivariada"

if st.sidebar.button("Análise de Mudança", key="change_btn", use_container_width=True):
    st.session_state['current_page'] = "model_change"

# Navegação baseada no estado
if st.session_state['current_page'] == "main":
    st.header("Upload de modelo")
    st.write("Estatísticas descritivas e visualização de dados.")

    model_1 = st.file_uploader("Faça upload do arquivo (.csv ou .parquet)", type=["csv", "parquet"])
    if model_1 is not None:
        # Verificar a extensão do arquivo
        file_extension = model_1.name.split('.')[-1].lower()
        
        try:
            if file_extension == 'csv':
                st.session_state['df'] = pd.read_csv(model_1)
            elif file_extension == 'parquet':
                st.session_state['df'] = pd.read_parquet(model_1)
                
            # Exibir informação sobre o formato do arquivo carregado
            st.info(f"Arquivo {model_1.name} carregado com sucesso ({file_extension.upper()})")
        except Exception as e:
            st.error(f"Erro ao carregar o arquivo: {str(e)}")
            st.session_state['df'] = None

    if st.button("Resetar dados"):
        reset_data()

    if st.session_state['df'] is not None:
        df = st.session_state['df']
        st.success("Arquivo carregado com sucesso!")
        
        st.subheader("Visualização 3D dos Pontos")
        
        # Colunas para seleção das coordenadas
        coord_col1, coord_col2, coord_col3 = st.columns(3)
        
        with coord_col1:
            x_coord = st.selectbox(
                "Selecione a coordenada X",
                options=df.columns,
                key="x_coord_select"
            )
        
        with coord_col2:
            y_coord = st.selectbox(
                "Selecione a coordenada Y",
                options=df.columns,
                key="y_coord_select"
            )
        
        with coord_col3:
            z_coord = st.selectbox(
                "Selecione a coordenada Z",
                options=df.columns,
                key="z_coord_select"
            )
        
        # Seleção da variável para colorir os pontos
        color_var = st.selectbox(
            "Selecione a variável para colorir os pontos",
            options=df.select_dtypes(include=['number']).columns,  # apenas variáveis numéricas
            key="color_var_select"
        )
        
        # Criar scatter 3D com as colunas selecionadas
        fig_3d = px.scatter_3d(
            df,
            x=x_coord,
            y=y_coord,
            z=z_coord,
            color=color_var,
            color_continuous_scale='RdBu_r',  # escala de cores: azul (frio/baixo) para vermelho (quente/alto)
            title="Distribuição Espacial dos Pontos",
            labels={
                x_coord: f'Coordenada X ({x_coord})',
                y_coord: f'Coordenada Y ({y_coord})',
                z_coord: f'Coordenada Z ({z_coord})',
                color_var: f'Valor de {color_var}'
            }
        )
        
        # Ajustar layout para melhor visualização
        fig_3d.update_layout(
            scene=dict(
                aspectmode='data',  # preserva a escala real dos dados
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=600
        )
        
        st.plotly_chart(fig_3d, use_container_width=True)
        
        st.subheader("Prévia dos dados")
        st.dataframe(df.head(), use_container_width=True)
        
        st.subheader("Estatísticas descritivas")
        st.write(df.describe())

elif st.session_state['current_page'] == "geostats":
    if st.session_state['subpage'] == "univariada":
        st.header("Análise Exploratória - Estatística Univariada")

        # Verifica se o DataFrame está carregado
        if st.session_state['df'] is not None:
            df = st.session_state['df']

            # Identificar variáveis categóricas e contínuas
            variaveis_categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()
            variaveis_continuas = df.select_dtypes(include=['float64', 'float32']).columns.tolist()

            # Layout para seleção de variáveis
            select_col1, select_col2 = st.columns([1, 1])

            with select_col1:
                # Seleção da variável contínua para análise
                coluna = st.selectbox(
                    "Selecione a variável contínua para análise:",
                    options=variaveis_continuas,
                    key="var_continua_select"
                )

            with select_col2:
                # Filtro por variável categórica
                var_categorica = None
                valor_escolhido = None
                if variaveis_categoricas:
                    var_categorica = st.selectbox(
                        "Filtrar por variável categórica:",
                        ["Nenhum filtro"] + variaveis_categoricas,
                        key="var_cat_uni"
                    )

            # Aplicar filtro se selecionado
            if var_categorica and var_categorica != "Nenhum filtro":
                valores_unicos = df[var_categorica].unique().tolist()
                valor_escolhido = st.selectbox(
                    f"Valor de {var_categorica}:",
                    valores_unicos,
                    key="valor_cat_uni"
                )
                df_filtrado = df[df[var_categorica] == valor_escolhido]
                st.info(f"Filtro aplicado: {var_categorica} = {valor_escolhido}")
            else:
                df_filtrado = df.copy()

            # Dashboard de gráficos
            st.write(f"**Variável selecionada:** `{coluna}`")
            
            dash_col1, dash_col2 = st.columns(2)

            with dash_col1:
                st.subheader("Histograma")
                fig1 = px.histogram(df_filtrado, x=coluna, nbins=30, opacity=0.7, 
                                title=f"Histograma de {coluna}")
                fig1.update_layout(
                    xaxis_title=coluna,
                    yaxis_title="Frequência",
                    bargap=0.1
                )
                st.plotly_chart(fig1, use_container_width=True)

            with dash_col2:
                st.subheader("Boxplot")
                fig2 = px.box(df_filtrado, x=coluna, points=False,
                           title=f"Boxplot de {coluna}")
                fig2.update_layout(
                    xaxis_title=coluna,
                    yaxis_title=coluna
                )
                st.plotly_chart(fig2, use_container_width=True)

        else:
            st.warning("Faça upload do arquivo na página principal para começar.")
    elif st.session_state['subpage'] == "multivariada":
        st.header("Análise Exploratória - Estatística Multivariada")
        
        # Verifica se o DataFrame está carregado
        if st.session_state['df'] is not None:
            df = st.session_state['df']
            
            # Identificar variáveis categóricas e contínuas
            variaveis_categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()
            variaveis_continuas = df.select_dtypes(include=['float64', 'float32']).columns.tolist()
            
            # Filtro por variável categórica
            if variaveis_categoricas:
                st.subheader("Filtro de Dados")
                filter_col1, filter_col2 = st.columns([1, 1])
                
                with filter_col1:
                    var_categorica = st.selectbox(
                        "Filtrar por variável categórica:",
                        ["Nenhum filtro"] + variaveis_categoricas,
                        key="var_cat_multi"
                    )
                
                with filter_col2:
                    if var_categorica != "Nenhum filtro":
                        valores_unicos = df[var_categorica].unique().tolist()
                        valor_filtro = st.selectbox(
                            f"Valor de {var_categorica}:",
                            valores_unicos,
                            key="valor_cat_multi"
                        )
                        # Aplicar filtro
                        df_filtrado = df[df[var_categorica] == valor_filtro]
                        st.info(f"Filtro aplicado: {var_categorica} = {valor_filtro}")
                    else:
                        df_filtrado = df.copy()
            else:
                df_filtrado = df.copy()
            
            # Layout em duas colunas para os gráficos
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Crossplot")
                # Seleção das variáveis para o crossplot
                var_x = st.selectbox(
                    "Variável X:",
                    options=variaveis_continuas,
                    key="var_x_multi"
                )
                var_y = st.selectbox(
                    "Variável Y:",
                    options=[col for col in variaveis_continuas if col != var_x],
                    key="var_y_multi"
                )
                
                # Opções de transformação
                opt_col1, opt_col2 = st.columns(2)
                with opt_col1:
                    use_log_x = st.checkbox("Log10 escala X", key="log_x_multi")
                    same_bounds = st.checkbox("Mesmos limites nos eixos", key="same_bounds_multi")
                with opt_col2:
                    use_log_y = st.checkbox("Log10 escala Y", key="log_y_multi")
                    swap_vars = st.checkbox("Trocar variáveis", key="swap_vars_multi")

                # Preparar dados para o crossplot
                plot_data = df.copy()
                x_var = var_y if swap_vars else var_x
                y_var = var_x if swap_vars else var_y

                if use_log_x:
                    plot_data = plot_data[plot_data[x_var] > 0]
                if use_log_y:
                    plot_data = plot_data[plot_data[y_var] > 0]

                # Criar crossplot
                fig_cross = px.scatter(
                    df_filtrado,  # Usando dados filtrados
                    x=x_var,
                    y=y_var,
                    title=f"Crossplot: {x_var} vs {y_var}",
                    trendline="ols"
                )

                # Aplicar transformações log se selecionadas
                if use_log_x:
                    fig_cross.update_xaxes(type="log")
                if use_log_y:
                    fig_cross.update_yaxes(type="log")

                # Aplicar mesmos limites se selecionado
                if same_bounds:
                    all_values = pd.concat([plot_data[x_var], plot_data[y_var]])
                    min_val = all_values.min()
                    max_val = all_values.max()
                    fig_cross.update_xaxes(range=[min_val, max_val])
                    fig_cross.update_yaxes(range=[min_val, max_val])

                # Calcular e mostrar correlação
                corr = plot_data[x_var].corr(plot_data[y_var])
                st.write(f"Correlação: {corr:.3f}")
                
                st.plotly_chart(fig_cross, use_container_width=True)

            with col2:
                st.subheader("Matriz de Correlação")
                # Seleção de variáveis para a matriz de correlação
                variaveis_matriz = st.multiselect(
                    "Selecione as variáveis para a matriz de correlação:",
                    options=variaveis_continuas,
                    default=list(variaveis_continuas[:4]) if len(variaveis_continuas) >= 4 else list(variaveis_continuas)
                )
                
                if len(variaveis_matriz) >= 2:
                    # Matriz de correlação usando dados filtrados
                    corr_matrix = df_filtrado[variaveis_matriz].corr()
                    
                    # Criar heatmap com plotly
                    fig_corr = px.imshow(
                        corr_matrix,
                        labels=dict(color="Correlação"),
                        color_continuous_scale="RdBu_r",
                        aspect="auto"
                    )
                    fig_corr.update_layout(
                        title="Matriz de Correlação das Variáveis Selecionadas",
                        height=500
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.warning("Selecione pelo menos duas variáveis para a matriz de correlação.")

    # Verifica se o DataFrame está carregado
    if st.session_state['df'] is not None:
        df = st.session_state['df']
    
    else:
        st.warning("Faça upload do arquivo .csv na página principal para começar.")

elif st.session_state['current_page'] == "model_change":
    st.header("Análise de Mudança de Modelo")
    st.write("Conteúdo para análise de mudança entre modelos.")


