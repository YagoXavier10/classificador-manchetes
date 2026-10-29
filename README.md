<p align="center">
  <img src="./assets/banner.svg" width="100%" alt="Classificador de Manchetes · machine learning no navegador" />
</p>

<p align="center">
  <a href="https://yagoxavier10.github.io/classificador-manchetes/"><img src="https://img.shields.io/badge/🤖_Testar_o_modelo-ao_vivo-7b3fe4?style=for-the-badge&labelColor=0d0b2e" alt="Testar o modelo" /></a>
  <a href="https://github.com/YagoXavier10/classificador-manchetes/actions/workflows/retrain.yml"><img src="https://github.com/YagoXavier10/classificador-manchetes/actions/workflows/retrain.yml/badge.svg" alt="Retreinar modelo" /></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-0d0b2e?style=for-the-badge&logo=python&logoColor=ffd43b" alt="Python" />
  <img src="https://img.shields.io/badge/scikit--learn-0d0b2e?style=for-the-badge&logo=scikitlearn&logoColor=f7931e" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/JavaScript-0d0b2e?style=for-the-badge&logo=javascript&logoColor=f7df1e" alt="JavaScript" />
  <img src="https://img.shields.io/badge/GitHub_Actions-0d0b2e?style=for-the-badge&logo=githubactions&logoColor=2088ff" alt="GitHub Actions" />
</p>

Modelo de **machine learning** que lê uma manchete de notícia e descobre o tema: Inteligência Artificial, Tecnologia,
Ciência de Dados, Economia, Política, Fintechs, Futebol, Motores ou Jiu-Jitsu.
O modelo roda **direto no navegador**, sem servidor, e é **retreinado toda semana** pelo GitHub Actions com notícias novas.

## 📸 Prévia

<p align="center">
  <a href="https://yagoxavier10.github.io/classificador-manchetes/"><img src="./assets/preview.png" width="90%" alt="Prévia do site" /></a>
</p>

## 📊 Resultados

Números do primeiro treino. A versão sempre atualizada fica em [`metrics.json`](metrics.json) e no [site](https://yagoxavier10.github.io/classificador-manchetes/):

- **Milhares de manchetes** reais coletadas do Google News, e o dataset cresce toda semana
- **~95% de acerto** em manchetes separadas para teste (20% que o modelo nunca viu no treino)
- Tema mais fácil: **Motores** (99%) · mais difícil: **Tecnologia** (88%), que se mistura com IA e Fintechs

> Os rótulos vêm da própria busca: uma notícia achada buscando "Selic OR Copom" vira "Economia".
> É um jeito barato de rotular milhares de exemplos, mas o acerto fica um pouco otimista,
> porque as palavras da busca aparecem nas manchetes.

## ⚙️ Como funciona

```text
Google News ──► collect.py ──► headlines.csv ──► train.py ──► model.json ──► app.js (navegador)
   (RSS)        coleta e        dataset           TF-IDF +      pesos do        mesma conta
                rotula          acumulado         Reg. Logística modelo          em JavaScript
```

1. **Coleta** · [`scripts/collect.py`](scripts/collect.py) busca manchetes de 9 temas no RSS do Google News e soma ao dataset.
2. **Texto → números** · cada manchete vira um vetor **TF-IDF** com palavras e pares de palavras, sem acentos e sem palavras vazias.
3. **Treino** · [`scripts/train.py`](scripts/train.py) treina uma **Regressão Logística** (scikit-learn) e mede o acerto num conjunto de teste.
4. **Exportação** · o vocabulário, os pesos IDF e os coeficientes vão para [`model.json`](model.json).
5. **No navegador** · [`app.js`](app.js) refaz a mesma conta em JavaScript e mostra as probabilidades e as palavras que mais pesaram.
6. **Toda segunda** · [`retrain.yml`](.github/workflows/retrain.yml) coleta notícias novas, retreina e publica sozinho.

## 💻 Rodar localmente

```bash
pip install -r requirements.txt
python scripts/collect.py      # baixa manchetes novas
python scripts/train.py        # treina e gera model.json e metrics.json
python -m http.server 8000     # abra http://localhost:8000
```

## 🗂️ Arquivos

| Arquivo | O que faz |
| :-- | :-- |
| `scripts/collect.py` | Coleta e rotula manchetes |
| `scripts/train.py` | Treina, avalia e exporta o modelo |
| `scripts/common.py` | Pré-processamento (o mesmo do `app.js`) |
| `data/headlines.csv` | Dataset de manchetes |
| `model.json` / `metrics.json` | Modelo exportado e métricas |
| `index.html` / `style.css` / `app.js` | Site que roda o modelo |

## 📄 Licença

Distribuído sob a licença MIT. Veja o arquivo [LICENSE](LICENSE).
