# MLAM — Demanda de bicicletas e mobilidade urbana

Esta entrega investiga uma pergunta simples e útil para uma operação de mobilidade: **as condições do dia ajudam a antecipar a procura por bicicletas compartilhadas?** A resposta foi construída com uma base pública real, análises estatísticas reproduzíveis e uma leitura cuidadosa das limitações.

## Integrantes

- Diogo Chiaradia Santos — RM 570246
- Gustavo Torres de Oliveira — RM 572952
- Lucas Furquim Lima — RM 568690
- Rafael Laprega Gontijo Magalhaes — RM 561975

## Estrutura

```text
mlam_humanizado/
├── analysis.py                         # código completo e reproduzível
├── requirements.txt
├── data/day.csv                        # base usada na análise
├── data/Readme.txt                     # documentação original da base
├── figures/                             # gráficos gerados pelo código
├── report/relatorio_estatistico_mobilidade.pdf
└── report/resultados.json               # números calculados, não estimados
```

## Fonte e escolha dos dados

Foi usado o **Bike Sharing Dataset**, do UCI Machine Learning Repository: <https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset>. A base contém registros diários de 2011 e 2012, com clima e quantidade de aluguéis. O arquivo original também identifica Fanaee-T e Gama como autores e informa o DOI `10.24432/C5W894`.

Esta não é uma base criada artificialmente. O arquivo `day.csv` foi mantido dentro de `data/` para que os resultados possam ser conferidos.

## O que foi feito

1. Probabilidade normal para a temperatura normalizada.
2. Dois testes unilaterais para a média, com `n=20` e `α=6%`, e `n=25` e `α=9%`.
3. Análise bivariada entre temperatura e total de aluguéis: dispersão, covariância, Pearson, reta, R², teste F e previsão.
4. Regressão linear múltipla com temperatura, umidade e vento, incluindo matrizes, R², teste F e previsão com três explicativas.
5. Regressão logística para classificar dias de alta demanda, usando o total acima ou igual à mediana como classe positiva.
6. Relatório em PDF com interpretações em linguagem direta, sem afirmar causalidade onde os dados só mostram associação.

## Como reproduzir

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python analysis.py
```

O script recria os gráficos, `report/resultados.json` e `report/relatorio_estatistico_mobilidade.pdf`.

## Leitura dos resultados

A correlação entre temperatura e demanda foi positiva (`r ≈ 0,627`) e o modelo simples explicou cerca de `39,4%` da variação. Com temperatura, umidade e vento, o R² chegou a aproximadamente `46,1%`. A regressão logística alcançou aproximadamente `77,6%` de acurácia no conjunto de teste.

Esses resultados não são uma promessa de demanda futura: faltam variáveis como chuva, feriados, dia útil, estação, eventos e tendência temporal. Para uso operacional, seria necessário validar o modelo em períodos futuros e monitorar mudanças de comportamento.
