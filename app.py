# -*- coding: utf-8 -*-
"""
NelFIT Suite Unificada (2D & 3D) - Versão Web para Browser
Convertido de Tkinter para Streamlit
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from scipy.optimize import curve_fit, differential_evolution
from scipy.stats import t  # <-- CORRIGIDO: Importação explícita do 't' de Student
from sympy import sympify, lambdify, symbols

# Configuração global da página Web
st.set_page_config(
    page_title="NELFIT Suite (v1.6 Web)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS personalizada
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1, h2, h3 { color: #1E3A8A; }
    .stButton>button { width: 100%; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)


# =============================================================================
# MÓDULO 2D
# =============================================================================
def render_modulo_2d():
    PRESETS_2D = {
        "Linear": "a*x + b",
        "Quadrático": "a*x**2 + b*x + c",
        "Exponencial": "a*exp(b*x)",
        "Potência": "a*x**b",
        "Logarítmico": "a*log(x) + b",
        "Senoide": "a*sin(b*x) + c",
        "Langmuir": "(a*b*x)/(1+(b*x))",
        "Bosch": "((16*x**2)+(18*a*(1-x)**2)+(b*c*x*(1-x)))/((x**2)+(a*(1-x)**2)+(c*x*(1-x)))"
    }

    if '2d_df' not in st.session_state: st.session_state['2d_df'] = None
    if '2d_resultados' not in st.session_state: st.session_state['2d_resultados'] = None

    col_lateral, col_graficos, col_tabelas = st.columns([1.2, 1.8, 1.2])

    with col_lateral:
        st.header("⚙️ Painel de Controlo 2D")
        
        st.subheader("1. Dados & Eixos")
        file_2d = st.file_uploader("Carregar Ficheiro (XLSX, CSV)", type=["csv", "xlsx"], key="file_2d")
        col_x_name = st.text_input("Nome da Coluna X:", value="x", key="col_x_2d")
        col_y_name = st.text_input("Nome da Coluna Y:", value="y", key="col_y_2d")

        if file_2d:
            try:
                if file_2d.name.endswith('.csv'):
                    df = pd.read_csv(file_2d)
                else:
                    df = pd.read_excel(file_2d)
                df.columns = [str(c).strip() for c in df.columns]
                st.session_state['2d_df'] = df
                st.success("Dados carregados com sucesso!")
            except Exception as e:
                st.error(f"Erro ao ler ficheiro: {e}")

        st.subheader("2. Equação & Modelo")
        preset_sel = st.selectbox("Modelos Predefinidos:", list(PRESETS_2D.keys()), key="preset_2d")
        formula_input = st.text_input("Equação Matemática (f(x)):", value=PRESETS_2D[preset_sel], key="formula_2d")
        
        st.subheader("3. Configurações do Ajuste")
        metodo = st.selectbox("Algoritmo de Otimização:", ["Levenberg-Marquardt (curve_fit)", "Evolução Diferencial (Global)"], key="metodo_2d")
        p0_raw = st.text_input("Palpites Iniciais / Limites (p0 ou min,max):", value="1.0, 1.0", key="p0_2d")
        
        conf_level = st.slider("Nível de Confiança (%):", min_value=80.0, max_value=99.9, value=95.0, step=0.5, key="conf_2d")
        
        usar_bootstrap = st.checkbox("Executar Bootstrap", value=False, key="boot_check_2d")
        n_boot = st.number_input("Nº de Simulações Bootstrap:", min_value=10, max_value=5000, value=200, step=50, disabled=not usar_bootstrap, key="n_boot_2d")

        btn_executar = st.button("🚀 EXECUTAR AJUSTE 2D", type="primary", key="btn_2d")

    if btn_executar:
        df = st.session_state['2d_df']
        if df is None:
            st.error("Por favor, carregue primeiro um ficheiro de dados.")
            return

        if col_x_name not in df.columns or col_y_name not in df.columns:
            st.error(f"Colunas não encontradas. Disponíveis: {list(df.columns)}")
            return

        try:
            df_clean = df[[col_x_name, col_y_name]].dropna().astype(float)
            X = df_clean[col_x_name].values
            Y = df_clean[col_y_name].values

            formula_python = formula_input.replace('^', '**')
            expr = sympify(formula_python)
            free_vars = expr.free_symbols
            param_names = sorted([str(s) for s in free_vars if str(s) != 'x'])
            
            sym_x = symbols('x')
            sym_params = [symbols(p) for p in param_names]
            func_lambd = lambdify([sym_x] + sym_params, expr, modules=['numpy', 'sympy'])

            p0_parsed = [float(x.strip()) for x in p0_raw.split(',') if x.strip()]

            if "Levenberg-Marquardt" in metodo:
                if len(p0_parsed) != len(param_names):
                    p0_parsed = [1.0] * len(param_names)
                popt, pcov = curve_fit(func_lambd, X, Y, p0=p0_parsed)
            else:
                bounds = []
                if len(p0_parsed) == len(param_names) * 2:
                    for i in range(0, len(p0_parsed), 2):
                        bounds.append((p0_parsed[i], p0_parsed[i+1]))
                else:
                    bounds = [(-10.0, 10.0)] * len(param_names)
                
                def obj_func(p):
                    return np.sum((Y - func_lambd(X, *p))**2)
                
                res_de = differential_evolution(obj_func, bounds)
                popt = res_de.x
                try:
                    _, pcov = curve_fit(func_lambd, X, Y, p0=popt)
                except:
                    pcov = None

            Y_pred = func_lambd(X, *popt)
            residuos = Y - Y_pred
            ss_res = np.sum(residuos**2)
            ss_tot = np.sum((Y - np.mean(Y))**2)
            r_sq = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
            
            n = len(X)
            p = len(popt)
            dof = max(1, n - p)
            r_sq_adj = 1 - ((1 - r_sq) * (n - 1) / dof)

            perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.zeros_like(popt)
            alpha = 1.0 - (conf_level / 100.0)
            
            # Aqui funciona corretamente porque 't' já está definido no topo
            t_val = t.ppf(1.0 - alpha / 2.0, dof)
            
            ci_lower = popt - t_val * perr
            ci_upper = popt + t_val * perr

            boot_errors = None
            if usar_bootstrap:
                boot_estimates = []
                for _ in range(n_boot):
                    idx = np.random.choice(n, size=n, replace=True)
                    X_b, Y_b = X[idx], Y[idx]
                    try:
                        popt_b, _ = curve_fit(func_lambd, X_b, Y_b, p0=popt, maxfev=1000)
                        boot_estimates.append(popt_b)
                    except:
                        continue
                if boot_estimates:
                    boot_errors = np.std(boot_estimates, axis=0)

            st.session_state['2d_resultados'] = {
                'X': X, 'Y': Y, 'Y_pred': Y_pred, 'residuos': residuos,
                'param_names': param_names, 'popt': popt, 'perr': perr,
                'ci_lower': ci_lower, 'ci_upper': ci_upper, 'boot_errors': boot_errors,
                'r_sq': r_sq, 'r_sq_adj': r_sq_adj, 'formula': formula_input,
                'func_lambd': func_lambd
            }
            st.toast("Ajuste 2D concluído!", icon="✅")
        except Exception as e:
            st.error(f"Erro matemático no ajuste 2D: {e}")

    res = st.session_state['2d_resultados']
    if res is not None:
        with col_graficos:
            st.subheader("📊 Visualização Gráfica (2D)")
            
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            ax1.scatter(res['X'], res['Y'], color='#DC2626', alpha=0.7, label='Dados Originais', edgecolors='k')
            x_line = np.linspace(np.min(res['X']), np.max(res['X']), 300)
            y_line = res['func_lambd'](x_line, *res['popt'])
            ax1.plot(x_line, y_line, color='#1E40AF', linewidth=2, label='Curva Ajustada')
            ax1.grid(True, linestyle=":", alpha=0.6)
            ax1.legend()
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots(figsize=(6, 2.2))
            ax2.scatter(res['X'], res['residuos'], color='#7C3AED', alpha=0.6, edgecolors='k')
            ax2.axhline(0, color='red', linestyle='--', linewidth=1)
            ax2.set_title("Distribuição de Resíduos")
            ax2.grid(True, linestyle=":", alpha=0.5)
            st.pyplot(fig2)

        with col_tabelas:
            st.subheader("📋 Métricas & Coeficientes")
            st.metric("Coeficiente R²", f"{res['r_sq']:.6f}")
            st.metric("R² Ajustado", f"{res['r_sq_adj']:.6f}")
            
            st.markdown("**Parâmetros Estimados:**")
            data_tabela = {
                "Parâmetro": res['param_names'],
                "Valor Estimado": [f"{v:.6f}" for v in res['popt']],
                "Erro Padrão": [f"{e:.6f}" for e in res['perr']],
                f"LCI ({conf_level}%)": [f"{l:.6f}" for l in res['ci_lower']],
                f"UCI ({conf_level}%)": [f"{u:.6f}" for u in res['ci_upper']]
            }
            if res['boot_errors'] is not None:
                data_tabela["Erro Bootstrap"] = [f"{b:.6f}" for b in res['boot_errors']]
                
            st.dataframe(pd.DataFrame(data_tabela), hide_index=True)


# =============================================================================
# MÓDULO 3D
# =============================================================================
def render_modulo_3d():
    PRESETS_3D = {
        "Plano Linear 3D": "a*x + b*y + c",
        "Parabolóide": "a*x**2 + b*y**2 + c",
        "Interação Sinérgica": "a*x*y + b*x + c*y + d",
        "Gaussiano 2D": "a*exp(-((x-b)**2 + (y-c)**2)/(2*d**2))"
    }

    if '3d_df' not in st.session_state: st.session_state['3d_df'] = None
    if '3d_resultados' not in st.session_state: st.session_state['3d_resultados'] = None

    col_lateral, col_graficos, col_tabelas = st.columns([1.2, 1.8, 1.2])

    with col_lateral:
        st.header("⚙️ Painel de Controlo 3D")
        
        st.subheader("1. Dados & Eixos Planos")
        file_3d = st.file_uploader("Carregar Ficheiro (XLSX, CSV)", type=["csv", "xlsx"], key="file_3d")
        col_x_name = st.text_input("Nome da Coluna X:", value="x", key="col_x_3d")
        col_y_name = st.text_input("Nome da Coluna Y:", value="y", key="col_y_3d")
        col_z_name = st.text_input("Nome da Coluna Z (Dependente):", value="z", key="col_z_3d")

        if file_3d:
            try:
                if file_3d.name.endswith('.csv'):
                    df = pd.read_csv(file_3d)
                else:
                    df = pd.read_excel(file_3d)
                df.columns = [str(c).strip() for c in df.columns]
                st.session_state['3d_df'] = df
                st.success("Dados 3D carregados com sucesso!")
            except Exception as e:
                st.error(f"Erro ao ler ficheiro 3D: {e}")

        st.subheader("2. Equação Espacial")
        preset_sel = st.selectbox("Modelos Predefinidos:", list(PRESETS_3D.keys()), key="preset_3d")
        formula_input = st.text_input("Equação f(x, y):", value=PRESETS_3D[preset_sel], key="formula_3d")
        
        st.subheader("3. Parâmetros")
        p0_raw = st.text_input("Valores Iniciais p0:", value="1.0, 1.0, 1.0", key="p0_3d")
        conf_level = st.slider("Nível de Confiança (%):", min_value=80.0, max_value=99.9, value=95.0, step=0.5, key="conf_3d")

        btn_executar = st.button("🚀 EXECUTAR AJUSTE 3D", type="primary", key="btn_3d")

    if btn_executar:
        df = st.session_state['3d_df']
        if df is None:
            st.error("Por favor, carregue primeiro um ficheiro de dados 3D.")
            return

        if not {col_x_name, col_y_name, col_z_name}.issubset(df.columns):
            st.error("Verifique as colunas do ficheiro.")
            return

        try:
            df_clean = df[[col_x_name, col_y_name, col_z_name]].dropna().astype(float)
            X = df_clean[col_x_name].values
            Y = df_clean[col_y_name].values
            Z = df_clean[col_z_name].values

            formula_python = formula_input.replace('^', '**')
            expr = sympify(formula_python)
            free_vars = expr.free_symbols
            param_names = sorted([str(s) for s in free_vars if str(s) not in ['x', 'y']])
            
            sym_x, sym_y = symbols('x y')
            sym_params = [symbols(p) for p in param_names]
            
            func_lambd = lambdify([(sym_x, sym_y)] + sym_params, expr, modules=['numpy', 'sympy'])

            p0_parsed = [float(x.strip()) for x in p0_raw.split(',') if x.strip()]
            if len(p0_parsed) != len(param_names):
                p0_parsed = [1.0] * len(param_names)

            popt, pcov = curve_fit(func_lambd, (X, Y), Z, p0=p0_parsed)

            Z_pred = func_lambd((X, Y), *popt)
            residuos = Z - Z_pred
            ss_res = np.sum(residuos**2)
            ss_tot = np.sum((Z - np.mean(Z))**2)
            r_sq = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
            
            n = len(Z)
            p = len(popt)
            dof = max(1, n - p)
            r_sq_adj = 1 - ((1 - r_sq) * (n - 1) / dof)
            
            perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.zeros_like(popt)
            alpha = 1.0 - (conf_level / 100.0)
            
            # CORRIGIDO: Agora encontra a distribuição t importada corretamente
            t_val = t.ppf(1.0 - alpha / 2.0, dof)
            
            ci_lower = popt - t_val * perr
            ci_upper = popt + t_val * perr

            st.session_state['3d_resultados'] = {
                'X': X, 'Y': Y, 'Z': Z, 'Z_pred': Z_pred, 'residuos': residuos,
                'param_names': param_names, 'popt': popt, 'perr': perr,
                'ci_lower': ci_lower, 'ci_upper': ci_upper, 'r_sq': r_sq, 'r_sq_adj': r_sq_adj,
                'func_lambd': func_lambd, 'formula': formula_input
            }
            st.toast("Ajuste de Superfície 3D concluído!", icon="✅")
        except Exception as e:
            st.error(f"Erro matemático no ajuste 3D: {e}")

    res = st.session_state['3d_resultados']
    if res is not None:
        with col_graficos:
            st.subheader("📊 Gráfico de Superfície Regressiva (3D)")
            
            fig = plt.figure(figsize=(6, 5))
            # CORRIGIDO: Inicialização limpa e universal da projeção 3D do Matplotlib
            ax = fig.add_subplot(111, projection='3d')
            
            ax.scatter(res['X'], res['Y'], res['Z'], color='red', s=25, label='Pontos Experimentais', edgecolors='k')
            
            x_grid = np.linspace(np.min(res['X']), np.max(res['X']), 40)
            y_grid = np.linspace(np.min(res['Y']), np.max(res['Y']), 40)
            X_m, Y_m = np.meshgrid(x_grid, y_grid)
            Z_m = res['func_lambd']((X_m, Y_m), *res['popt'])
            
            surf = ax.plot_surface(X_m, Y_m, Z_m, cmap='viridis', alpha=0.6, edgecolor='none')
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Z Ajustado')
            
            ax.set_xlabel('Eixo X')
            ax.set_ylabel('Eixo Y')
            ax.set_zlabel('Eixo Z')
            ax.legend()
            st.pyplot(fig)

        with col_tabelas:
            st.subheader("📋 Métricas Globais 3D")
            st.metric("R² Espacial", f"{res['r_sq']:.6f}")
            st.metric("R² Ajustado", f"{res['r_sq_adj']:.6f}")

            st.markdown("**Coeficientes da Superfície:**")
            df_para_3d = pd.DataFrame({
                "Parâmetro": res['param_names'],
                "Estimativa": [f"{v:.6f}" for v in res['popt']],
                "Erro Padrão": [f"{e:.6f}" for e in res['perr']],
                f"LCI ({conf_level}%)": [f"{l:.6f}" for l in res['ci_lower']],
                f"UCI ({conf_level}%)": [f"{u:.6f}" for u in res['ci_upper']]
            })
            st.dataframe(df_para_3d, hide_index=True)


# =============================================================================
# ESTRUTURA PRINCIPAL DE ABAS
# =============================================================================
def main():
    st.title("🧪 NELFIT Suite Unificada (v1.6 Web)")
    st.caption("Ajuste de Curvas e Superfícies Científicas Avançadas — Executado diretamente no seu Navegador")

    aba1, aba2 = st.tabs(["📈 Módulo de Ajuste 2D", "🌐 Módulo de Superfícies 3D"])

    with aba1:
        render_modulo_2d()

    with aba2:
        render_modulo_3d()

if __name__ == "__main__":
    main()
