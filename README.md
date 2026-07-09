# Entrega — Mineração de Dados

**Título:** Morfologia polínica e classificação por aprendizado de máquina de espécies de *Cololobus* (Asteraceae, Vernonieae)

Esta pasta reúne os itens exigidos para a entrega do projeto de Mineração de Dados: o dataset, o código de análise, o relatório em LaTeX/PDF e as tabelas de features analisadas.

## Estrutura

```
Entrega Mineração de Dados/
├── README.md                     # este arquivo
├── relatorio/                    # artigo em LaTeX (template Springer Nature sn-jnl)
│   ├── relatorio_cololobus.pdf   # >>> PDF COMPILADO (entrega final) <<<
│   ├── relatorio_cololobus.tex   # relatório COMPLEMENTADO (EDA + Mineração + ML)
│   ├── tabela_features_analisadas.tex   # 14 features: Kruskal-Wallis + importância ML
│   ├── tabela_ml_resultados.tex         # comparação de modelos e métricas por espécie
│   ├── tabela_cololobus_morfometria.tex # tabela morfométrica (eixos por espécie)
│   ├── cololobus-bibliography.bib
│   ├── sn-jnl.cls                # classe do template
│   └── sn-mathphys-num.bst       # estilo de bibliografia
├── images/                       # figuras usadas na compilação do PDF
└── codigo/                       # pipeline Python reprodutível (auto-contido)
    ├── requirements.txt
    ├── data/tables/polen_features_analyze.csv   # dataset analisado (89 grãos, 14 features)
    ├── scripts/run_analysis.py                  # EDA + ML → figuras e relatórios .md
    └── src/pollen_ml/                           # pacote: config, preprocess, plots, ML
```

## O que cada script faz

### Entregável: `codigo/scripts/run_analysis.py`

Script **único de execução** da análise. Carrega o CSV, roda EDA + ML e grava figuras e relatórios auxiliares.

| Função interna | O que faz | Item do enunciado |
|----------------|-----------|-------------------|
| `run_eda()` | Pré-processamento, estatística, Kruskal-Wallis, Shapiro-Wilk, correlação, PCA, histogramas, box/violin plots, scatter plots | **2) Mineração exploratória** |
| `run_ml()` | Classificação (Dummy, RF, SVM), GridSearchCV, validação cruzada, matriz de confusão, importância de variáveis, K-Means + ARI | **3) Aprendizado de máquina** |
| `write_eda_report()` / `write_ml_report()` | Gera `docs/resultados_eda.md` e `docs/resultados_ml.md` com tabelas numéricas | Apoio ao **4) Relatório** |

Comando:

```bash
cd codigo && PYTHONPATH=src python scripts/run_analysis.py
```

### Pacote de apoio: `codigo/src/pollen_ml/`

| Módulo | Papel |
|--------|-------|
| `config.py` | Lista das 14 features, caminho do CSV, função `load_features()` |
| `preprocess.py` | Limpeza (`valid=1`), imputação por vista, `StandardScaler` para ML/PCA |
| `eda_plots.py` | Todas as figuras exploratórias (histogramas, violinos, scatter, PCA, heatmap) |
| `pca_pipelines.py` | Compara três níveis de pré-processamento no PCA (exploratório extra) |
| `ml_plots.py` | Gráficos de comparação entre modelos (F1, acurácia por fold) |
| `ml_interpretation.py` | Matriz normalizada, SHAP, permutation importance, erros por par de espécies |
| `filenames.py` | Utilitário para parsear nomes `.czi` (não usado na análise atual; só equatorial) |

### Fora da entrega: `DOC_ML/scripts/generate_latex_morpho_table.py`

Gera `tabela_cololobus_morfometria.tex` a partir do CSV. É ferramenta de **redação do artigo**, não parte da pipeline de mineração/ML. Por isso ficou no repositório principal, separado de `codigo/`.

### Relatório: `relatorio/`

Arquivos LaTeX **já prontos** (não gerados pelo Python na entrega):

- `relatorio_cololobus.tex` + `.pdf` — artigo completo
- `tabela_features_analisadas.tex` — 14 features (Kruskal-Wallis + importância ML)
- `tabela_ml_resultados.tex` — comparação de modelos
- `tabela_cololobus_morfometria.tex` — eixos polar/equatorial por espécie

## Reproduzir a análise

```bash
cd codigo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python scripts/run_analysis.py
```

Isso recria as figuras (em `codigo/outputs/figures/`) e os relatórios `.md` de resultados (em `codigo/docs/`), além do resumo numérico `codigo/outputs/ml_summary.txt`.

## Compilar o relatório

```bash
cd relatorio
pdflatex relatorio_cololobus.tex
bibtex   relatorio_cololobus
pdflatex relatorio_cololobus.tex
pdflatex relatorio_cololobus.tex
```

## Figuras

O **PDF já compilado** está em `relatorio/relatorio_cololobus.pdf`. As figuras usadas ficam em `images/` (referenciadas via `\graphicspath{{../images/}...}`). Para regerá-las, execute `run_analysis.py`: os PNGs aparecem em `codigo/outputs/figures/` (a prancha `Prancha_polen.png` é a única imagem de microscopia, não gerada pelo código).

## Principais resultados

- Kruskal-Wallis: **11 de 14** features diferem significativamente entre espécies (α = 0,05).
- Modelo final: **SVM RBF** (C=1, γ=0,1) — **F1-macro = 0,72** e acurácia = 0,72 em validação cruzada 5-fold (linha de base *Dummy* = 0,21).
- *C. hatschbachii* é a espécie mais bem classificada (F1 = 0,88); os erros concentram-se entre *C. longiangustatus*, *C. rupestris* e *C. ruschianus*.
- Variáveis mais discriminantes: comprimento do espinho, eixo polar, solidez e largura da ectoabertura.
