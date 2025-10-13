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
    .stButton > button, .stButton button, div[data-testid="stHorizontalBlock"] button, button[kind="secondary"] {
        background-color: #FFE600 !important;
        color: #1A1A24 !important;
        border: none !important;
        font-weight: bold !important;
    }
    
    .stButton > button:hover, .stButton button:hover, div[data-testid="stHorizontalBlock"] button:hover, button[kind="secondary"]:hover {
        background-color: #E6CF00 !important;
        color: #1A1A24 !important;
    }
    
    /* Forçar cor do texto em todos os botões */
    button[data-testid="baseButton-secondary"], 
    .stButton > button > div, 
    .stButton button > div,
    .stButton button p,
    .stButton button span {
        color: #1A1A24 !important;
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

            st.subheader("Gráficos de Probabilidade")

            # Layout para opções dos gráficos de probabilidade
            prob_col1, prob_col2 = st.columns(2)

            with prob_col1:
                plot_type = st.selectbox(
                    "Tipo de gráfico:",
                    ["QQ-Plot", "PP-Plot", "NP-Plot"],
                    key="prob_plot_type"
                )
                
                distribution = st.selectbox(
                    "Distribuição teórica:",
                    ["Gaussian", "Lognormal", "Uniform", "Gamma", "Exponential"],
                    key="theoretical_dist"
                )

            with prob_col2:
                n_quantiles = st.slider(
                    "Número de quantis:",
                    min_value=10,
                    max_value=100,
                    value=20,
                    step=10,
                    key="n_quantiles"
                )
                
                use_log = st.checkbox("Usar escala Log10", key="prob_log_scale")

            # Preparar dados para os gráficos
            import numpy as np
            from scipy import stats

            def prepare_probability_plot(data, plot_type, distribution, n_quantiles):
                # Remover valores nulos
                data = data.dropna()
                
                if distribution == "Gaussian":
                    theoretical_dist = stats.norm(loc=data.mean(), scale=data.std())
                elif distribution == "Lognormal":
                    theoretical_dist = stats.lognorm(s=data.std(), loc=0, scale=np.exp(data.mean()))
                elif distribution == "Uniform":
                    theoretical_dist = stats.uniform(loc=data.min(), scale=data.max()-data.min())
                elif distribution == "Gamma":
                    # Estimativa dos parâmetros da distribuição gamma
                    alpha_est = (data.mean() ** 2) / (data.var())
                    beta_est = data.mean() / data.var()
                    theoretical_dist = stats.gamma(a=alpha_est, scale=1/beta_est)
                else:  # Exponential
                    theoretical_dist = stats.expon(scale=1/data.mean())
                
                # Calcular quantis
                empirical_quantiles = np.percentile(data, np.linspace(0, 100, n_quantiles))
                theoretical_quantiles = theoretical_dist.ppf(np.linspace(0.01, 0.99, n_quantiles))
                
                if plot_type == "QQ-Plot":
                    return empirical_quantiles, theoretical_quantiles
                elif plot_type == "PP-Plot":
                    emp_cdf = np.linspace(0, 1, n_quantiles)
                    theo_cdf = theoretical_dist.cdf(empirical_quantiles)
                    return emp_cdf, theo_cdf
                else:  # NP-Plot
                    sorted_data = np.sort(data)
                    prob = np.linspace(0, 1, len(sorted_data))
                    return sorted_data, stats.norm.ppf(prob)

            # Criar o gráfico
            if not df_filtrado[coluna].empty:
                x_vals, y_vals = prepare_probability_plot(df_filtrado[coluna], plot_type, distribution, n_quantiles)
                
                fig_prob = px.scatter(
                    x=x_vals,
                    y=y_vals,
                    title=f"{plot_type} - {coluna} vs {distribution}",
                    labels={
                        "x": "Valores Empíricos" if plot_type != "PP-Plot" else "Probabilidade Empírica",
                        "y": "Valores Teóricos" if plot_type != "PP-Plot" else "Probabilidade Teórica"
                    }
                )
                
                # Adicionar linha de referência (primeira bissetriz)
                min_val = min(x_vals.min(), y_vals.min())
                max_val = max(x_vals.max(), y_vals.max())
                fig_prob.add_scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    line=dict(color="red", dash="dash"),
                    name="Referência"
                )
                
                # Aplicar escala logarítmica se selecionado
                if use_log:
                    if plot_type != "PP-Plot":  # PP-Plot sempre usa escala linear
                        fig_prob.update_layout(
                            xaxis_type="log",
                            yaxis_type="log"
                        )
                
                # Ajustar layout
                fig_prob.update_layout(
                    showlegend=True,
                    width=800,
                    height=500
                )
                
                st.plotly_chart(fig_prob, use_container_width=True)
                
                # Realizar teste Chi-quadrado
                if st.checkbox("Realizar teste Chi-quadrado", key="chi2_test"):
                    n_classes = st.slider(
                        "Número de classes para teste Chi-quadrado:",
                        min_value=5,
                        max_value=30,
                        value=10,
                        key="n_classes_chi2"
                    )

                    # Definir a distribuição teórica novamente para o teste
                    if distribution == "Gaussian":
                        theoretical_dist = stats.norm(loc=df_filtrado[coluna].mean(), scale=df_filtrado[coluna].std())
                    elif distribution == "Lognormal":
                        theoretical_dist = stats.lognorm(s=df_filtrado[coluna].std(), loc=0, scale=np.exp(df_filtrado[coluna].mean()))
                    elif distribution == "Uniform":
                        theoretical_dist = stats.uniform(loc=df_filtrado[coluna].min(), scale=df_filtrado[coluna].max()-df_filtrado[coluna].min())
                    elif distribution == "Gamma":
                        alpha_est = (df_filtrado[coluna].mean() ** 2) / (df_filtrado[coluna].var())
                        beta_est = df_filtrado[coluna].mean() / df_filtrado[coluna].var()
                        theoretical_dist = stats.gamma(a=alpha_est, scale=1/beta_est)
                    else:  # Exponential
                        theoretical_dist = stats.expon(scale=1/df_filtrado[coluna].mean())

                    # Calcular teste Chi-quadrado
                    observed, bins = np.histogram(df_filtrado[coluna], bins=n_classes)
                    expected = len(df_filtrado[coluna]) * np.diff(
                        theoretical_dist.cdf(bins)
                    )

                    chi2_stat, p_value = stats.chisquare(observed, expected)

                    st.write(f"""
                    **Resultados do teste Chi-quadrado:**
                    - Estatística Chi-quadrado: {chi2_stat:.2f}
                    - Valor-p: {p_value:.4f}
                    """)
            else:
                st.warning(f"Nenhum dado disponível para {coluna} após a filtragem.")

            # Swath Plot
            st.subheader("Swath Plot")
            
            # Opções do Swath Plot
            swath_col1, swath_col2 = st.columns(2)
            
            with swath_col1:
                # Seleção da direção
                direcao = st.selectbox(
                    "Direção do Swath:",
                    options=["X (Leste-Oeste)", "Y (Norte-Sul)", "Z (Elevação)"],
                    key="swath_direction"
                )
                
                # Tamanho das fatias
                slice_size = st.number_input(
                    "Tamanho das fatias:",
                    min_value=1.0,
                    value=10.0,
                    step=1.0,
                    key="slice_size"
                )
                
                # Tipo de visualização
                plot_type = st.selectbox(
                    "Tipo de visualização:",
                    options=["Boxplot", "Tendência"],
                    key="plot_type"
                )
            
            with swath_col2:
                # Cor dos boxplots
                box_color = st.color_picker(
                    "Cor dos boxplots:",
                    value="#1f77b4",
                    key="box_color"
                )
                
                # Mostrar número de amostras
                show_samples = st.selectbox(
                    "Exibir número de amostras:",
                    options=["Não exibir", "Texto", "Histograma"],
                    key="show_samples"
                )
                
                # Mostrar outliers (apenas para boxplot)
                if plot_type == "Boxplot":
                    show_outliers = st.checkbox("Mostrar outliers", value=True, key="show_outliers")

            # Encontrar colunas de coordenadas
            coord_cols = {
                'X': next((col for col in df_filtrado.columns if 'XC' in col.upper()), None),
                'Y': next((col for col in df_filtrado.columns if 'YC' in col.upper()), None),
                'Z': next((col for col in df_filtrado.columns if 'ZC' in col.upper()), None)
            }
            
            # Criar o Swath Plot
            if direcao == "X (Leste-Oeste)":
                coord = coord_cols['X']
            elif direcao == "Y (Norte-Sul)":
                coord = coord_cols['Y']
            else:
                coord = coord_cols['Z']

            # Calcular as fatias
            if coord is not None:
                min_coord = df_filtrado[coord].min()
                max_coord = df_filtrado[coord].max()
                n_slices = int((max_coord - min_coord) / slice_size) + 1
                
                # Criar os bins
                df_filtrado['slice'] = pd.cut(df_filtrado[coord], 
                                            bins=n_slices, 
                                            labels=[f"Slice {i+1}" for i in range(n_slices)])
                
                # Calcular estatísticas por fatia
                stats = df_filtrado.groupby('slice')[coluna].agg(['mean', 'std', 'count'])
                slice_centers = [(i + 0.5) * slice_size + min_coord for i in range(n_slices)]
                
                # Criar o gráfico
                if plot_type == "Boxplot":
                    fig_swath = px.box(df_filtrado, x='slice', y=coluna,
                                     title=f"Swath Plot - {direcao} - {coluna}",
                                     points='outliers' if show_outliers else False,
                                     color_discrete_sequence=[box_color])
                    
                    # Adicionar linha conectando as médias
                    fig_swath.add_scatter(x=stats.index, y=stats['mean'],
                                        mode='lines+markers',
                                        name='Média',
                                        line=dict(color='red', width=2))
                    
                else:  # Tendência
                    fig_swath = px.line(x=slice_centers, y=stats['mean'],
                                      title=f"Swath Plot - {direcao} - {coluna}")
                    
                    # Adicionar bandas de desvio padrão
                    fig_swath.add_scatter(x=slice_centers, 
                                        y=stats['mean'] + stats['std'],
                                        mode='lines',
                                        name='+1 Std Dev',
                                        line=dict(dash='dash'))
                    fig_swath.add_scatter(x=slice_centers, 
                                        y=stats['mean'] - stats['std'],
                                        mode='lines',
                                        name='-1 Std Dev',
                                        line=dict(dash='dash'),
                                        fill='tonexty')

                # Ajustar layout
                fig_swath.update_layout(
                    xaxis_title=coord,
                    yaxis_title=coluna,
                    showlegend=True
                )
                
                # Mostrar número de amostras
                if show_samples != "Não exibir":
                    if show_samples == "Texto":
                        for i, count in enumerate(stats['count']):
                            fig_swath.add_annotation(x=i, y=stats['mean'].max(),
                                                   text=f"n={count}",
                                                   showarrow=False)
                    else:  # Histograma
                        fig_swath.add_bar(x=stats.index, y=stats['count'],
                                        name='Número de amostras',
                                        yaxis='y2')
                        fig_swath.update_layout(
                            yaxis2=dict(title='Número de amostras',
                                      overlaying='y',
                                      side='right')
                        )
                
                st.plotly_chart(fig_swath, use_container_width=True)
            else:
                if direcao == "X (Leste-Oeste)":
                    st.warning("Nenhuma coluna contendo 'XC' foi encontrada no conjunto de dados.")
                elif direcao == "Y (Norte-Sul)":
                    st.warning("Nenhuma coluna contendo 'YC' foi encontrada no conjunto de dados.")
                else:
                    st.warning("Nenhuma coluna contendo 'ZC' foi encontrada no conjunto de dados.")

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


