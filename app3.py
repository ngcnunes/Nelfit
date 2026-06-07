
# -*- coding: utf-8 -*-
"""
NelFIT Suite Unificada (v1.6) - Versão Web Streamlit Fiel à Original
Recuperação total de métricas estatísticas, limites e algoritmos globais.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from scipy.optimize import curve_fit, differential_evolution
from scipy.stats import t, qmc  # Incluindo amostragem Quase-Monte Carlo
from sympy import sympify, lambdify, symbols

# Configuração global da página Web
st.set_page_config(
    page_title="NELFIT Suite (v1.6 Web Completo)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS personalizada
st.markdown("""
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
    h1, h2, h3 { color: #0d47a1; }
    .stButton>button { width: 100%; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# Inicializar estados de sessão se não existirem (essencial para persistência igual ao Tkinter)
if '2d_df' not in st.session_state: st.session_state['2d_df'] = None
if '2d_pontos_ativos' not in st.session_state: st.session_state['2d_pontos_ativos'] = None
if '2d_resultados' not in st.session_state: st.session_state['2d_resultados'] = None

if '3d_df' not in st.session_state: st.session_state['3d_df'] = None
if '3d_pontos_ativos' not in st.session_state: st.session_state['3d_pontos_ativos'] = None
if '3d_resultados' not in st.session_state: st.session_state['3d_resultados'] = None


# --- FUNÇÕES AUXILIARES DE SUPORTE MATEMÁTICO (Fieis ao Tkinter) ---
def parse_bounds(bounds_str, n_params):
    if not bounds_str.strip(): return None
    try:
        pares = [b.strip() for b in bounds_str.split(',')]
        if len(pares) != n_params: raise ValueError()
        lower, upper = [], []
        for par in pares:
            lo_str, up_str = [s.strip().lower() for s in par.split(':', 1)]
            lo = -np.inf if lo_str in ('-inf', '') else float(lo_str)
            up = np.inf if up_str in ('inf', '+inf', '') else float(up_str)
            lower.append(lo)
            upper.append(up)
        return (lower, upper)
    except:
        raise ValueError(f"Formato inválido. Esperados {n_params} pares 'min:max' separados por vírgulas.")

def gerar_palpites(n_params, n_amostras, bounds, p0_base, metodo):
    lo = np.full(n_params, -np.inf)
    up = np.full(n_params, np.inf)
    if bounds is not None:
        lo[:] = bounds[0]
        up[:] = bounds[1]
    for i in range(n_params):
        base = p0_base[i] if p0_base[i] != 0 else 1.0
        escala = max(abs(base) * 15, 2.0)
        if not np.isfinite(lo[i]): lo[i] = base - escala
        if not np.isfinite(up[i]): up[i] = base + escala
    if metodo == "Latin Hypercube":
        sampler = qmc.LatinHypercube(d=n_params, seed=42)
        palpites = qmc.scale(sampler.random(n=n_amostras), lo, up)
    else:
        palpites = np.random.uniform(lo, up, size=(n_amostras, n_params))
    return np.vstack([np.array(p0_base), palpites])

def procura_global(func_num, p0_base, bounds, n_params, x_mat, y_mat, metodo, n_tentativas):
    if metodo == "Differential Evolution":
        lo = [-100.0] * n_params
        up = [100.0] * n_params
        if bounds is not None:
            lo[:] = bounds[0]
            up[:] = bounds[1]
        def ssr_fun(params):
            try: return np.sum((y_mat - func_num(x_mat, *params)) ** 2)
            except: return 1e20
        result = differential_evolution(ssr_fun, list(zip(lo, up)), seed=42, maxiter=50, tol=1e-7, polish=True)
        try: _, pcov = curve_fit(func_num, x_mat, y_mat, p0=result.x, maxfev=1000)
        except: pcov = np.eye(n_params) * 1e-6
        return result.x, pcov
    
    palpites = gerar_palpites(n_params, n_tentativas, bounds, p0_base, metodo)
    melhor_popt, melhor_pcov, melhor_ssr = None, None, np.inf
    for p0_tent in palpites:
        try:
            if bounds is None: popt, pcov = curve_fit(func_num, x_mat, y_mat, p0=p0_tent, maxfev=2000)
            else: popt, pcov = curve_fit(func_num, x_mat, y_mat, p0=p0_tent, bounds=bounds, maxfev=2000)
            ssr = np.sum((y_mat - func_num(x_mat, *popt)) ** 2)
            if ssr < melhor_ssr:
                melhor_ssr, melhor_popt, melhor_pcov = ssr, popt, pcov
        except: continue
    if melhor_popt is None: raise RuntimeError("A otimização global local falhou em convergir.")
    return melhor_popt, melhor_pcov


# =============================================================================
# MÓDULO 2D (COMPLETO)
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

    col_lateral, col_graficos, col_resultados = st.columns([1.1, 1.8, 1.1])

    with col_lateral:
        st.header("1. Dados & Eixos")
        file_2d = st.file_uploader("Carregar Ficheiro (XLSX)", type=["xlsx"], key="f2d")
        
        if file_2d:
            try:
                df = pd.read_csv(file_2d) if file_2d.name.endswith('.csv') else pd.read_excel(file_2d)
                df.columns = [str(c).strip().lower() for c in df.columns]
                if 'x' in df.columns and 'y' in df.columns:
                    if st.session_state['2d_df'] is None or len(st.session_state['2d_df']) != len(df):
                        st.session_state['2d_df'] = df
                        st.session_state['2d_pontos_ativos'] = np.ones(len(df), dtype=bool)
                else:
                    st.error("O ficheiro deve conter colunas 'x' e 'y'.")
            except Exception as e: st.error(f"Erro: {e}")

        lbl_x = st.text_input("Título Eixo X:", value="Eixo X")
        lbl_y = st.text_input("Título Eixo Y:", value="Eixo Y")

        # Gerenciamento Interativo de Outliers via Tabela 
        if st.session_state['2d_df'] is not None:
            st.markdown("💡 **Controlo de Ativação de Pontos:**")
            n_tot = len(st.session_state['2d_df'])
            n_at = int(np.sum(st.session_state['2d_pontos_ativos']))
            st.caption(f"✔ {n_at} ativos de {n_tot} totais. Desmarque para excluir:")
            
            df_edit = st.session_state['2d_df'].copy()
            df_edit['Ativo'] = st.session_state['2d_pontos_ativos']
            
            with st.expander("Ver/Excluir Pontos Experimentais"):
                for idx, row in df_edit.iterrows():
                    val = st.checkbox(f"Ponto [{idx}]: x={row['x']:.4g}, y={row['y']:.4g}", value=bool(row['Ativo']), key=f"p2d_{idx}")
                    st.session_state['2d_pontos_ativos'][idx] = val

        st.header("2. Equação & Modelo")
        preset_sel = st.selectbox("Modelos Predefinidos 2D:", list(PRESETS_2D.keys()))
        formula_input = st.text_input("Função (variável 'x'):", value=PRESETS_2D[preset_sel])
        
        p0_raw = st.text_input("Palpites Iniciais p0 (opcional, sep. por vírgula):", value="")
        bounds_raw = st.text_input("Limites (min:max) ex: -10:10, 0:5 (Opcional):", value="")

        st.subheader("Opções Avançadas")
        usar_global = st.checkbox("Ativar procura global (Multistart)", value=False)
        metodo_global = st.selectbox("Algoritmo pontos iniciais:", ["Latin Hypercube", "Aleatório", "Differential Evolution"], disabled=not usar_global)
        tentativas_global = st.number_input("Nº Tentativas:", min_value=5, max_value=500, value=50, step=10, disabled=not usar_global)

        btn_executar = st.button("🚀 EXECUTAR AJUSTE 2D", type="primary")

    # --- PROCESSAMENTO DO AJUSTE 2D  ---
    if btn_executar:
        if st.session_state['2d_df'] is None:
            st.error("Carregue dados primeiro.")
            return

        ativos = st.session_state['2d_pontos_ativos']
        if np.sum(ativos) < 3:
            st.error("Dados ativos insuficientes para realizar a regressão (mínimo 3).")
            return

        try:
            X_clean = st.session_state['2d_df']['x'].values[ativos].astype(float)
            Y_clean = st.session_state['2d_df']['y'].values[ativos].astype(float)

            expr = sympify(formula_input.replace('^', '**'))
            param_names = sorted([str(s) for s in expr.free_symbols if str(s) != 'x'])
            n_params = len(param_names)

            p0_parsed = [float(x.strip()) for x in p0_raw.split(',') if x.strip()] if p0_raw else [1.0] * n_params
            if len(p0_parsed) != n_params: p0_parsed = [1.0] * n_params
            
            bounds_parsed = parse_bounds(bounds_raw, n_params) if bounds_raw else None

            sym_x = symbols('x')
            sym_params = [symbols(p) for p in param_names]
            func_lambd = lambdify([sym_x] + sym_params, expr, modules=['numpy', 'sympy'])

            def func_wrapper(x, *params): return func_lambd(x, *params)

            if usar_global:
                popt, pcov = procura_global(func_wrapper, p0_parsed, bounds_parsed, n_params, X_clean, Y_clean, metodo_global, tentativas_global)
                status_msg = f"✔ Otimização Global ({metodo_global}) Concluída."
            else:
                if bounds_parsed is None: popt, pcov = curve_fit(func_wrapper, X_clean, Y_clean, p0=p0_parsed, maxfev=5000)
                else: popt, pcov = curve_fit(func_wrapper, X_clean, Y_clean, p0=p0_parsed, bounds=bounds_parsed, maxfev=5000)
                status_msg = "✔ Ajuste Padrão Concluído."

            Y_pred = func_lambd(X_clean, *popt)
            residuos = Y_clean - Y_pred
            ssr = np.sum(residuos**2)
            sst = np.sum((Y_clean - np.mean(Y_clean))**2)
            
            n_dados = len(Y_clean)
            dof = n_dados - n_params
            
            r2 = 1.0 - (ssr / sst) if sst != 0 else 0.0
            r2_adj = 1.0 - ((1.0 - r2) * (n_dados - 1) / dof) if dof > 0 else 0.0
            r2_tablecurve = 1.0 - ((1.0 - r2) * (n_dados - 1) / (dof - 1)) if dof > 1 else 0.0
            syx = np.sqrt(ssr / dof) if dof > 0 else 0.0
            
            f_snedecor = ((sst - ssr) / (n_params - 1)) / (ssr / dof) if (dof > 0 and n_params > 1 and ssr != 0) else 0.0
            perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.zeros(n_params)
            
            p_values = []
            for i in range(n_params):
                if perr[i] != 0:
                    t_stat = popt[i] / perr[i]
                    p_val = 2 * (1 - t.cdf(abs(t_stat), df=max(1, dof)))
                    p_values.append(p_val)
                else: p_values.append(np.nan)

            st.session_state['2d_resultados'] = {
                'X_all': st.session_state['2d_df']['x'].values, 'Y_all': st.session_state['2d_df']['y'].values,
                'X_clean': X_clean, 'Y_clean': Y_clean, 'Y_pred': Y_pred, 'residuos': residuos,
                'param_names': param_names, 'popt': popt, 'perr': perr, 'p_values': p_values,
                'r2': r2, 'r2_adj': r2_adj, 'r2_tc': r2_tablecurve, 'syx': syx, 'f_snedecor': f_snedecor,
                'formula': formula_input, 'func_lambd': func_lambd, 'status_msg': status_msg, 'lbl_x': lbl_x, 'lbl_y': lbl_y
            }
            st.toast("Ajuste executado com sucesso!", icon="✅")
        except Exception as e:
            st.error(f"O ajuste falhou ou não convergiu: {e}")

    res = st.session_state['2d_resultados']
    if res is not None:
        with col_graficos:
            st.subheader("📊 Visualização de Gráficos Integrados")
            aba_graf1, aba_graf2 = st.tabs(["📈 Gráfico de Ajuste 2D", "📉 Análise Detalhada de Resíduos"])
            ativos = st.session_state['2d_pontos_ativos']
            excluidos = ~ativos

            with aba_graf1:
                fig1, ax1 = plt.subplots(figsize=(6, 4))
                if np.any(ativos):
                    ax1.scatter(res['X_all'][ativos], res['Y_all'][ativos], color='red', s=40, alpha=0.8, edgecolors='k', label='Dados Ativos', zorder=5)
                if np.any(excluidos):
                    ax1.scatter(res['X_all'][excluidos], res['Y_all'][excluidos], color='#bdc3c7', marker='X', s=50, alpha=0.6, edgecolors='#7f8c8d', label='Excluídos (Outliers)', zorder=4)
                
                x_line = np.linspace(np.min(res['X_all']), np.max(res['X_all']), 300)
                try:
                    y_line = res['func_lambd'](x_line, *res['popt'])
                    ax1.plot(x_line, y_line, color='blue', linewidth=2, label='Curva de Ajuste')
                except: pass
                ax1.set_xlabel(res['lbl_x'], fontsize=9)
                ax1.set_ylabel(res['lbl_y'], fontsize=9)
                ax1.grid(True, linestyle=':', alpha=0.5)
                ax1.legend(fontsize=8)
                st.pyplot(fig1)

            with aba_graf2:
                fig2, (ax2_top, ax2_bot) = plt.subplots(2, 1, figsize=(6, 4.5), gridspec_kw={'height_ratios': [5, 4]})
                ax2_top.scatter(res['X_clean'], res['residuos'], c=res['residuos'], cmap='coolwarm', s=35, edgecolors='k')
                ax2_top.axhline(0, color='black', linestyle='--', linewidth=1)
                ax2_top.set_ylabel("Resíduo", fontsize=8)
                ax2_top.grid(True, linestyle=':', alpha=0.5)
                
                n_bins = max(5, int(np.sqrt(len(res['residuos']))))
                ax2_bot.hist(res['residuos'], bins=n_bins, color='#3498db', edgecolor='black', alpha=0.8)
                ax2_bot.axvline(0, color='red', linestyle='--', linewidth=1)
                ax2_bot.set_xlabel("Resíduo", fontsize=8)
                ax2_bot.set_ylabel("Frequência", fontsize=8)
                ax2_bot.grid(True, linestyle=':', alpha=0.5)
                
                plt.tight_layout()
                st.pyplot(fig2)

        with col_resultados:
            st.subheader("📋 4. Resultados")
            st.markdown("**Parâmetros Estimados:**")
            p_val_strings = [f"{p:.4f}" if (not np.isnan(p) and p >= 0.001) else ("< 0.001" if p < 0.001 else "---") for p in res['p_values']]
            df_table = pd.DataFrame({
                "Par.": res['param_names'],
                "Estimativa": [f"{v:.5g}" for v in res['popt']],
                "Erro (±)": [f"{e:.3g}" for e in res['perr']],
                "p-value": p_val_strings
            })
            st.dataframe(df_table, hide_index=True)

            st.markdown("**Estatísticas de Ajuste:**")
            st.text(f"R²: {res['r2']:.6f}")
            st.text(f"R² Ajustado: {res['r2_adj']:.6f}")
            st.text(f"R² Adj (TableCurve): {res['r2_tc']:.6f}")
            st.text(f"Desvio Padrão (Sy.x): {res['syx']:.5g}")
            st.text(f"F-Snedecor: {res['f_snedecor']:.4f}")
            st.caption(res['status_msg'])

            report_data = f"Equacao: {res['formula']}\n\n" + df_table.to_csv(index=False) + \
                          f"\nEstatisticas:\nR2: {res['r2']:.6f}\nR2 Adj: {res['r2_adj']:.6f}\nSy.x: {res['syx']:.5g}\nF-Stat: {res['f_snedecor']:.4g}"
            st.download_button("💾 GRAVAR RESULTADOS (.CSV)", data=report_data, file_name="resultados_nelfit_2d.csv", mime="text/csv")


# =============================================================================
# MÓDULO 3D 
# =============================================================================
def render_modulo_3d():
    PRESETS_3D = {
        "Linear": "a*x + b*y + c",
        "Plano Quadrático 1": "a*x**2 + b*y + c",
        "Plano Quadrático 2": "a*x + b*y**2 + c",
        "Cúbico Simples": "a*x + b*y + c*x**2 + d*y**2",
        "Exponencial": "a*exp(b*x) + c*exp(d*y)",
        "Potência": "a*(x**b) + c*(y**d)",
        "Logarítmico": "a*log(x) + b*log(y) + c",
        "Senoide": "a*sin(b*x) + c*cos(d*y) + e"
    }

    col_lateral, col_graficos, col_resultados = st.columns([1.1, 1.8, 1.1])

    with col_lateral:
        st.header("1. Dados & Eixos ")
        file_3d = st.file_uploader("Carregar Ficheiro 3D (XLSX)", type=["xlsx"], key="f3d")
        
        if file_3d:
            try:
                df = pd.read_csv(file_3d) if file_3d.name.endswith('.csv') else pd.read_excel(file_3d)
                df.columns = [str(c).strip().lower() for c in df.columns]
                if {'x', 'y', 'z'}.issubset(df.columns):
                    if st.session_state['3d_df'] is None or len(st.session_state['3d_df']) != len(df):
                        st.session_state['3d_df'] = df
                        st.session_state['3d_pontos_ativos'] = np.ones(len(df), dtype=bool)
                else: 
                    st.error("O ficheiro deve conter as colunas obrigatórias 'x', 'y' e 'z'.")
            except Exception as e: 
                st.error(f"Erro ao ler ficheiro: {e}")

        lbl_x = st.text_input("Etiqueta Eixo X:", value="Eixo X", key="lbl_x_3d")
        lbl_y = st.text_input("Etiqueta Eixo Y:", value="Eixo Y", key="lbl_y_3d")
        lbl_z = st.text_input("Etiqueta Eixo Z:", value="Eixo Z", key="lbl_z_3d")

        # Gerenciamento Interativo de Outliers via Tabela (Corrigido para 3D)
        if st.session_state['3d_df'] is not None:
            st.markdown("💡 **Controlo de Ativação de Pontos:**")
            n_tot = len(st.session_state['3d_df'])
            n_at = int(np.sum(st.session_state['3d_pontos_ativos']))
            st.caption(f"✔ {n_at} ativos de {n_tot} totais. Desmarque para excluir:")
            
            df_edit = st.session_state['3d_df'].copy()
            df_edit['Ativo'] = st.session_state['3d_pontos_ativos']
            
            with st.expander("Ver/Excluir Pontos Experimentais"):
                for idx, row in df_edit.iterrows():
                    val = st.checkbox(f"Ponto [{idx}]: x={row['x']:.4g}, y={row['y']:.4g}, z={row['z']:.4g}", value=bool(row['Ativo']), key=f"p3d_{idx}")
                    st.session_state['3d_pontos_ativos'][idx] = val

        st.header("2. Equação & Modelo")
        preset_sel = st.selectbox("Modelos Predefinidos 3D:", list(PRESETS_3D.keys()))
        formula_input = st.text_input("Função f(x, y):", value=PRESETS_3D[preset_sel])
        
        p0_raw = st.text_input("Palpites p0  (opcional, sep. por vírgulas):", value="")
        bounds_raw = st.text_input("Limites (min:max) ex: -5:5, 0:inf (Opcional):", value="")

        st.subheader("Opções de Otimização")
        usar_global = st.checkbox("Ativar procura global 3D (Multistart)", value=False, key="global_3d")
        metodo_global = st.selectbox("Algoritmo pontos iniciais:", ["Latin Hypercube", "Aleatório", "Differential Evolution"], disabled=not usar_global, key="met_3d")
        tentativas_global = st.number_input("Nº de Tentativas Globais:", min_value=5, max_value=500, value=50, step=10, disabled=not usar_global, key="tent_3d")
        
        conf_level = st.slider("Nível de Confiança para IC (%):", min_value=80.0, max_value=99.9, value=95.0, step=0.5, key="conf_3d")

        btn_executar = st.button("🚀 EXECUTAR AJUSTE 3D", type="primary")

    # --- MOTOR DE PROCESSAMENTO MATEMÁTICO 3D ---
    if btn_executar:
        if st.session_state['3d_df'] is None:
            st.error("Por favor, carregue os dados 3D.")
            return

        ativos = st.session_state['3d_pontos_ativos']
        if np.sum(ativos) < 4:
            st.error("Dados ativos insuficientes para realizar a regressão espacial 3D (mínimo 4).")
            return

        try:
            # Filtragem exata dos pontos ativos no processamento (Corrigido)
            X_clean = st.session_state['3d_df']['x'].values[ativos].astype(float)
            Y_clean = st.session_state['3d_df']['y'].values[ativos].astype(float)
            Z_clean = st.session_state['3d_df']['z'].values[ativos].astype(float)

            expr = sympify(formula_input.replace('^', '**'))
            param_names = sorted([str(s) for s in expr.free_symbols if str(s) not in ['x', 'y']])
            n_params = len(param_names)

            p0_parsed = [float(x.strip()) for x in p0_raw.split(',') if x.strip()] if p0_raw else [1.0] * n_params
            if len(p0_parsed) != n_params: 
                p0_parsed = [1.0] * n_params
            
            bounds_parsed = parse_bounds(bounds_raw, n_params) if bounds_raw else None

            sym_x, sym_y = symbols('x y')
            sym_params = [symbols(p) for p in param_names]
            func_lambd = lambdify([(sym_x, sym_y)] + sym_params, expr, modules=['numpy', 'sympy'])

            def func_wrapper(xy_tuple, *params): 
                return func_lambd(xy_tuple, *params)

            if usar_global:
                popt, pcov = procura_global(func_wrapper, p0_parsed, bounds_parsed, n_params, (X_clean, Y_clean), Z_clean, metodo_global, tentativas_global)
                status_msg = f"✔ Otimização Global 3D ({metodo_global}) Concluída."
            else:
                if bounds_parsed is None: 
                    popt, pcov = curve_fit(func_wrapper, (X_clean, Y_clean), Z_clean, p0=p0_parsed, maxfev=5000)
                else: 
                    popt, pcov = curve_fit(func_wrapper, (X_clean, Y_clean), Z_clean, p0=p0_parsed, bounds=bounds_parsed, maxfev=5000)
                status_msg = "✔ Ajuste de Superfície Padrão Concluído."

            Z_pred = func_lambd((X_clean, Y_clean), *popt)
            residuos = Z_clean - Z_pred
            ssr = np.sum(residuos**2)
            sst = np.sum((Z_clean - np.mean(Z_clean))**2)
            
            n_dados = len(Z_clean)
            dof = n_dados - n_params
            
            r2 = 1.0 - (ssr / sst) if sst != 0 else 0.0
            r2_adj = 1.0 - ((1.0 - r2) * (n_dados - 1) / dof) if dof > 0 else 0.0
            r2_tablecurve = 1.0 - ((1.0 - r2) * (n_dados - 1) / (dof - 1)) if dof > 1 else 0.0
            syx = np.sqrt(ssr / dof) if dof > 0 else 0.0
            
            f_snedecor = ((sst - ssr) / (n_params - 1)) / (ssr / dof) if (dof > 0 and n_params > 1 and ssr != 0) else 0.0
            perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.zeros(n_params)
            
            alpha = 1.0 - (conf_level / 100.0)
            t_val = t.ppf(1.0 - alpha / 2.0, max(1, dof))
            
            ci_lower = popt - t_val * perr
            ci_upper = popt + t_val * perr
            
            p_values = []
            for i in range(n_params):
                if perr[i] != 0:
                    t_stat = popt[i] / perr[i]
                    p_val = 2 * (1 - t.cdf(abs(t_stat), df=max(1, dof)))
                    p_values.append(p_val)
                else: 
                    p_values.append(np.nan)

            st.session_state['3d_resultados'] = {
                'X_all': st.session_state['3d_df']['x'].values, 'Y_all': st.session_state['3d_df']['y'].values, 'Z_all': st.session_state['3d_df']['z'].values,
                'X_clean': X_clean, 'Y_clean': Y_clean, 'Z_clean': Z_clean, 'Z_pred': Z_pred, 'residuos': residuos,
                'param_names': param_names, 'popt': popt, 'perr': perr, 
                'ci_lower': ci_lower, 'ci_upper': ci_upper, 'p_values': p_values,
                'r2': r2, 'r2_adj': r2_adj, 'r2_tc': r2_tablecurve, 'syx': syx, 'f_snedecor': f_snedecor,
                'formula': formula_input, 'func_lambd': func_lambd, 'status_msg': status_msg,
                'lbl_x': lbl_x, 'lbl_y': lbl_y, 'lbl_z': lbl_z, 'conf_level': conf_level
            }
            st.toast("Ajuste 3D concluído!", icon="✅")
        except Exception as e: 
            st.error(f"Erro matemático no ajuste espacial 3D: {e}")

    # --- RENDERING DE GRÁFICOS E TABELAS DE RESULTADOS 3D ---
    res = st.session_state['3d_resultados']
    if res is not None:
        with col_graficos:
            st.subheader("📊 Visualização de Modelos")
            aba_graf1, aba_graf2 = st.tabs(["🌐 Superfície  3D", "📉  Resíduos"])
            ativos = st.session_state['3d_pontos_ativos']
            excluidos = ~ativos
            
            with aba_graf1:
                fig = plt.figure(figsize=(6, 5))
                ax = fig.add_subplot(111, projection='3d')
                
                if np.any(ativos):
                    ax.scatter(res['X_all'][ativos], res['Y_all'][ativos], res['Z_all'][ativos], color='red', s=25, label='Dados Ativos', edgecolors='k', alpha=0.9, zorder=5)
                if np.any(excluidos):
                    ax.scatter(res['X_all'][excluidos], res['Y_all'][excluidos], res['Z_all'][excluidos], color='#bdc3c7', marker='X', s=35, label='Excluídos', edgecolors='#7f8c8d', alpha=0.6, zorder=4)
                
                x_grid = np.linspace(np.min(res['X_all']), np.max(res['X_all']), 40)
                y_grid = np.linspace(np.min(res['Y_all']), np.max(res['Y_all']), 40)
                X_m, Y_m = np.meshgrid(x_grid, y_grid)
                try:
                    Z_m = res['func_lambd']((X_m, Y_m), *res['popt'])
                    surf = ax.plot_surface(X_m, Y_m, Z_m, cmap='viridis', alpha=0.6, edgecolor='none', zorder=1)
                    fig.colorbar(surf, ax=ax, shrink=0.4, aspect=10, label=f"{res['lbl_z']} Calculado")
                except: pass
                
                ax.set_xlabel(res['lbl_x'], fontsize=8)
                ax.set_ylabel(res['lbl_y'], fontsize=8)
                ax.set_zlabel(res['lbl_z'], fontsize=8)
                ax.legend(fontsize=8)
                st.pyplot(fig)

            with aba_graf2:
                fig2, (ax2_top, ax2_bot) = plt.subplots(2, 1, figsize=(6, 4.5), gridspec_kw={'height_ratios': [5, 4]})
                ax2_top.scatter(res['Z_pred'], res['residuos'], c=res['residuos'], cmap='coolwarm', s=30, edgecolors='k', alpha=0.8)
                ax2_top.axhline(0, color='black', linestyle='--', linewidth=1)
                ax2_top.set_xlabel(f"{res['lbl_z']} Previsto", fontsize=8)
                ax2_top.set_ylabel("Resíduo", fontsize=8)
                ax2_top.grid(True, linestyle=':', alpha=0.5)
                
                n_bins = max(5, int(np.sqrt(len(res['residuos']))))
                ax2_bot.hist(res['residuos'], bins=n_bins, color='#e67e22', edgecolor='black', alpha=0.8)
                ax2_bot.axvline(0, color='red', linestyle='--', linewidth=1)
                ax2_bot.set_xlabel("Erro Residual", fontsize=8)
                ax2_bot.set_ylabel("Frequência", fontsize=8)
                ax2_bot.grid(True, linestyle=':', alpha=0.5)
                
                plt.tight_layout()
                st.pyplot(fig2)

        with col_resultados:
            st.subheader("📋 4. Resultados")
            st.markdown(f"**Coeficientes da Superfície ({res['conf_level']}% IC):**")
            p_val_strings = [f"{p:.4f}" if (not np.isnan(p) and p >= 0.001) else ("< 0.001" if p < 0.001 else "---") for p in res['p_values']]
            df_table_3d = pd.DataFrame({
                "Par.": res['param_names'],
                "Estimativa": [f"{v:.5g}" for v in res['popt']],
                "Erro (±)": [f"{e:.3g}" for e in res['perr']],
                "p-value": p_val_strings,
                "LCI": [f"{l:.4g}" for l in res['ci_lower']],
                "UCI": [f"{u:.4g}" for u in res['ci_upper']]
            })
            st.dataframe(df_table_3d, hide_index=True)

            st.markdown("**Estatísticas de Ajuste:**")
            st.text(f"R² : {res['r2']:.6f}")
            st.text(f"R² Ajustado: {res['r2_adj']:.6f}")
            st.text(f"R² Adj (TableCurve): {res['r2_tc']:.6f}")
            st.text(f"Desvio Padrão (Sy.x): {res['syx']:.5g}")
            st.text(f"F-Snedecor 3D: {res['f_snedecor']:.4f}")
            st.caption(res['status_msg'])

            report_data_3d = f"Equacao Superficie: {res['formula']}\n\n" + df_table_3d.to_csv(index=False) + \
                             f"\nEstatisticas 3D:\nR2: {res['r2']:.6f}\nR2 Adj: {res['r2_adj']:.6f}\nSy.x: {res['syx']:.5g}\nF-Stat: {res['f_snedecor']:.4f}"
            st.download_button("💾 GRAVAR RESULTADOS 3D (.CSV)", data=report_data_3d, file_name="resultados_nelfit_3d.csv", mime="text/csv")


# =============================================================================
# ESTRUTURA PRINCIPAL DE ABAS
# =============================================================================
def main():
    st.title("🧪 NELFIT  (v1.6 Web)")
    st.caption("Ajustes Matemáticos 2D & 3D")

    aba1, aba2 = st.tabs(["📈 Módulo de Ajuste 2D", "🌐 Módulo de Ajuste 3D"])

    with aba1: render_modulo_2d()
    with aba2: render_modulo_3d()

if __name__ == "__main__":
    main()