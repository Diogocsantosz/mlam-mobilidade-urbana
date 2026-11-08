"""Análise estatística da demanda diária de bicicletas.

Projeto MLAM — base pública real do UCI Machine Learning Repository.
O script gera as tabelas, gráficos e o relatório PDF sem depender de dados fictícios.
"""
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "day.csv"
FIG = ROOT / "figures"
OUT = ROOT / "report"
FIG.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
RANDOM_STATE = 42

def load_data():
    df = pd.read_csv(DATA)
    df["dteday"] = pd.to_datetime(df["dteday"])
    # nomes mais legíveis para o relatório
    return df.rename(columns={
        "temp": "temperatura_normalizada", "atemp": "sensacao_termica_normalizada",
        "hum": "umidade_normalizada", "windspeed": "vento_normalizado",
        "cnt": "alugueis_totais", "registered": "alugueis_cadastrados",
        "casual": "alugueis_avulsos"
    })

def savefig(name):
    path = FIG / name
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path

def fmt(x, digits=3):
    return f"{x:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")

def build_report(df, results):
    pdf = OUT / "relatorio_estatistico_mobilidade.pdf"
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, fontSize=18, leading=23, spaceAfter=16))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.5, leading=11))
    doc = SimpleDocTemplate(str(pdf), pagesize=A4, rightMargin=1.6*cm, leftMargin=1.6*cm, topMargin=1.5*cm, bottomMargin=1.5*cm)
    story = []
    P = lambda text, style="BodyText": story.append(Paragraph(text, styles[style]))
    story.append(Paragraph("Demanda de bicicletas e mobilidade urbana", styles["TitleCenter"]))
    P("<b>Relatório estatístico — MLAM</b><br/>Análise desenvolvida com uma base pública real do UCI Machine Learning Repository. A proposta é entender como condições climáticas se relacionam com o uso do sistema de bicicletas compartilhadas e se essas variáveis ajudam a identificar dias de alta demanda.")
    story.append(Spacer(1, 10))
    P("<b>Integrantes:</b> Diogo Chiaradia Santos (RM 570246); Gustavo Torres de Oliveira (RM 572952); Lucas Furquim Lima (RM 568690); Rafael Laprega Gontijo Magalhaes (RM 561975).")
    P("<b>Problema:</b> operadores de mobilidade precisam antecipar a procura para distribuir bicicletas e planejar a operação. A análise não afirma causalidade: ela mostra associações observadas na base.", "BodyText")
    P("<b>Base e tratamento.</b> Foram usados 731 registros diários, entre 2011 e 2012. A base contém temperatura, sensação térmica, umidade, vento e total de aluguéis. Não havia valores ausentes nas colunas utilizadas. As variáveis climáticas são normalizadas na própria base (0 a 1 aproximadamente), por isso as conclusões falam em aumento relativo, e não em graus Celsius.")
    P("Fonte: Fanaee-T e Gama, <i>Bike Sharing Dataset</i>, UCI Machine Learning Repository, DOI 10.24432/C5W894. Acesso em 08/11/2026. Arquivo original preservado em data/day.csv.")
    story.append(Spacer(1, 8))
    P("1. Probabilidade normal", "Heading1")
    P(f"Escolhemos a temperatura normalizada. Média = {fmt(results['mean_temp'])}; mediana = {fmt(results['median_temp'])}; desvio-padrão = {fmt(results['std_temp'])}. Sob a suposição normal, P(X &lt; média) = {fmt(results['p_less_mean']*100, 2)}%, um evento equiprovável. Para P(X &gt; mediana), a aproximação normal resultou em {fmt(results['p_greater_median']*100, 2)}%, também aproximadamente equiprovável, embora a mediana observada não seja exatamente igual à média.")
    P("2. Testes de hipótese para a média", "Heading1")
    P(f"Usamos como referência a média de toda a população observada e amostras aleatórias reproduzíveis. No teste unilateral à esquerda (n=20, α=6%), média amostral = {fmt(results['sample_a_mean'])}, z = {fmt(results['z_a'])}, p = {fmt(results['p_a'], 4)}: {results['decision_a']}. No teste unilateral à direita (n=25, α=9%), média amostral = {fmt(results['sample_b_mean'])}, z = {fmt(results['z_b'])}, p = {fmt(results['p_b'], 4)}: {results['decision_b']}. Como os testes usam a média da própria base como referência, eles funcionam como uma verificação didática, não como evidência de causalidade.")
    P("3. Análise bivariada", "Heading1")
    P(f"A relação entre temperatura e aluguéis totais foi positiva: covariância = {fmt(results['cov_temp_count'], 4)} e correlação de Pearson r = {fmt(results['r_temp_count'], 3)}. O modelo linear simples estimou {results['eq_simple']} e explicou R² = {fmt(results['r2_simple'], 3)} da variação observada. O teste F teve p = {fmt(results['f_pvalue_simple'], 5)}, indicando associação estatisticamente detectável neste conjunto. Para uma temperatura normalizada de 0,70, a previsão do modelo é de aproximadamente {fmt(results['prediction_simple'], 0)} aluguéis. Ainda assim, o gráfico mostra dispersão: temperatura sozinha não basta para planejar a operação.")
    story.append(Image(str(FIG / "01_dispersao_regressao_simples.png"), width=16*cm, height=9*cm))
    story.append(PageBreak())
    P("4. Regressão linear múltipla", "Heading1")
    P(f"Incluímos temperatura, umidade e vento para explicar os aluguéis totais. A equação estimada foi: {results['eq_multiple']}. O modelo alcançou R² = {fmt(results['r2_multiple'], 3)} (R² ajustado = {fmt(results['r2_adj_multiple'], 3)}) e teste F com p = {fmt(results['f_pvalue_multiple'], 5)}. Para um cenário com temperatura=0,70, umidade=0,55 e vento=0,20, a previsão é de {fmt(results['prediction_multiple'], 0)} aluguéis. O R² indica que ainda há fatores importantes fora do modelo, como estação, feriados, dias úteis e tendência temporal.")
    story.append(Image(str(FIG / "02_matriz_correlacao.png"), width=14*cm, height=10*cm))
    P("Matriz de covariância e correlação", "Heading2")
    cov = results["cov_table"]
    corr = results["corr_table"]
    data = [["Variável", *cov.columns]] + [[idx, *[fmt(v, 3) for v in cov.loc[idx]]] for idx in cov.index]
    t = Table(data, repeatRows=1, colWidths=[4.2*cm]*4)
    t.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#245a73")), ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("GRID", (0,0), (-1,-1), .25, colors.grey), ("FONTSIZE", (0,0), (-1,-1), 7)]))
    story.append(t)
    story.append(Spacer(1, 8))
    P("Matriz de correlação de Pearson", "Heading2")
    data_corr = [["Variável", *corr.columns]] + [[idx, *[fmt(v, 3) for v in corr.loc[idx]]] for idx in corr.index]
    tc = Table(data_corr, repeatRows=1, colWidths=[4.2*cm]*4)
    tc.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#4b8b8a")), ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("GRID", (0,0), (-1,-1), .25, colors.grey), ("FONTSIZE", (0,0), (-1,-1), 7)]))
    story.append(tc)
    P("Na matriz de correlação, temperatura apresenta a associação linear mais evidente com a demanda entre as variáveis climáticas usadas. Correlação não significa que o clima, sozinho, cause o aumento: calendário e comportamento também participam do resultado.", "Small")
    P("5. Regressão logística binária", "Heading1")
    P(f"Criamos a classe <b>alta demanda</b> quando os aluguéis totais ficaram acima ou iguais à mediana ({fmt(results['median_count'], 0)}). O modelo usa temperatura, umidade e vento. A acurácia no conjunto de teste foi {fmt(results['accuracy']*100, 2)}%. A matriz de confusão foi TN={results['cm'][0][0]}, FP={results['cm'][0][1]}, FN={results['cm'][1][0]} e TP={results['cm'][1][1]}. A classificação é útil como alerta operacional, mas não substitui uma previsão completa, pois o corte pela mediana é uma escolha analítica.")
    P("Equação em logito: " + results["logit_eq"])
    story.append(Image(str(FIG / "03_matriz_confusao.png"), width=9*cm, height=8*cm))
    P("6. Conclusão e limitações", "Heading1")
    P(f"Os dados sugerem que dias mais quentes tendem a ter mais aluguéis: r={fmt(results['r_temp_count'], 3)} e R² simples={fmt(results['r2_simple'], 3)}. Ao combinar temperatura, umidade e vento, o R² passa a {fmt(results['r2_multiple'], 3)}, mostrando ganho, mas também deixando claro que a maior parte da variação depende de outros fatores. A classificação de alta demanda atingiu {fmt(results['accuracy']*100, 2)}% de acurácia. O próximo passo seria incorporar feriados, dias úteis, estação, chuva e tendência temporal, além de validar o modelo em um período futuro. Os resultados são associações da base e não devem ser tratados como causalidade ou garantia de demanda.")
    doc.build(story)
    return pdf

def main():
    sns.set_theme(style="whitegrid", palette="deep")
    df = load_data()
    temp, count = df["temperatura_normalizada"], df["alugueis_totais"]
    results = {"n": len(df), "mean_temp": temp.mean(), "median_temp": temp.median(), "std_temp": temp.std(ddof=1)}
    results["p_less_mean"] = stats.norm.cdf(results["mean_temp"], results["mean_temp"], results["std_temp"])
    results["p_greater_median"] = 1 - stats.norm.cdf(results["median_temp"], results["mean_temp"], results["std_temp"])
    rng = np.random.default_rng(RANDOM_STATE)
    sa, sb = rng.choice(temp, 20, replace=False), rng.choice(temp, 25, replace=False)
    mu, sigma = temp.mean(), temp.std(ddof=1)
    for key, sample, alpha, side in [("a", sa, .06, "less"), ("b", sb, .09, "greater")]:
        z = (sample.mean()-mu)/(sigma/np.sqrt(len(sample)))
        p = stats.norm.cdf(z) if side == "less" else 1-stats.norm.cdf(z)
        results[f"sample_{key}_mean"], results[f"z_{key}"], results[f"p_{key}"] = sample.mean(), z, p
        results[f"decision_{key}"] = "rejeitamos H0" if p < alpha else "não rejeitamos H0"
    results["cov_temp_count"] = temp.cov(count); results["r_temp_count"] = temp.corr(count)
    X1 = sm.add_constant(temp); simple = sm.OLS(count, X1).fit()
    results.update(r2_simple=simple.rsquared, f_pvalue_simple=float(simple.f_pvalue), prediction_simple=float(simple.predict([1, .70])[0]))
    results["eq_simple"] = f"y = {simple.params.iloc[0]:.2f} + {simple.params.iloc[1]:.2f} × temperatura"
    plt.figure(figsize=(9, 5)); sns.regplot(data=df, x="temperatura_normalizada", y="alugueis_totais", scatter_kws={"alpha": .45}, line_kws={"color": "#d95f02"}); plt.title("Temperatura e demanda diária"); plt.xlabel("Temperatura normalizada"); plt.ylabel("Total de aluguéis"); savefig("01_dispersao_regressao_simples.png")
    cols = ["temperatura_normalizada", "umidade_normalizada", "vento_normalizado", "alugueis_totais"]
    results["cov_table"] = df[cols].cov(); results["corr_table"] = df[cols].corr()
    plt.figure(figsize=(8, 6)); sns.heatmap(results["corr_table"], annot=True, fmt=".2f", cmap="vlag", center=0); plt.title("Correlação de Pearson"); savefig("02_matriz_correlacao.png")
    features = ["temperatura_normalizada", "umidade_normalizada", "vento_normalizado"]
    Xm = sm.add_constant(df[features]); multiple = sm.OLS(count, Xm).fit()
    results.update(r2_multiple=multiple.rsquared, r2_adj_multiple=multiple.rsquared_adj, f_pvalue_multiple=float(multiple.f_pvalue), prediction_multiple=float(multiple.predict([1, .70, .55, .20])[0]))
    pms = multiple.params; results["eq_multiple"] = "y = " + " + ".join([f"({pms[i]:.2f} × {i})" for i in features]) + f" + ({pms['const']:.2f})"
    df["alta_demanda"] = (count >= count.median()).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(df[features], df["alta_demanda"], test_size=.25, random_state=RANDOM_STATE, stratify=df["alta_demanda"])
    clf = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(random_state=RANDOM_STATE))]); clf.fit(Xtr, ytr); pred = clf.predict(Xte)
    results["accuracy"] = accuracy_score(yte, pred); results["cm"] = confusion_matrix(yte, pred).tolist()
    coef = clf.named_steps["model"].coef_[0]; inter = clf.named_steps["model"].intercept_[0]
    results["logit_eq"] = f"logit(p) = {inter:.3f} + ({coef[0]:.3f})·temperatura + ({coef[1]:.3f})·umidade + ({coef[2]:.3f})·vento"
    plt.figure(figsize=(5, 4)); sns.heatmap(results["cm"], annot=True, fmt="d", cmap="Blues", cbar=False); plt.title("Matriz de confusão"); plt.xlabel("Predito"); plt.ylabel("Real"); savefig("03_matriz_confusao.png")
    results["median_count"] = count.median()
    serializable = {k: (v.to_dict() if isinstance(v, pd.DataFrame) else v) for k,v in results.items() if k not in ["cov_table", "corr_table"]}
    (OUT / "resultados.json").write_text(json.dumps(serializable, ensure_ascii=False, indent=2, default=lambda x: float(x)), encoding="utf-8")
    build_report(df, results)
    print(f"Relatório gerado: {OUT / 'relatorio_estatistico_mobilidade.pdf'}")

if __name__ == "__main__": main()
