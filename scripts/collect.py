"""Coleta manchetes do Google News por tema e soma ao dataset em data/headlines.csv.

Cada busca vira um rótulo: é assim que o modelo aprende sem ninguém rotular nada à mão.
"""
import csv
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote_plus

import requests

TOPICS = {
    "Inteligência Artificial": ['"inteligência artificial"', "ChatGPT OR OpenAI OR Anthropic OR Gemini", '"IA generativa" OR "modelo de IA"'],
    "Tecnologia": ["smartphone OR celular lançamento", "Apple OR Samsung OR Google tecnologia", "internet OR 5G OR software"],
    "Ciência de Dados": ['"ciência de dados" OR "cientista de dados"', '"machine learning" OR "big data"', '"análise de dados" OR "dados abertos"'],
    "Economia": ["Selic OR Copom OR inflação", "IPCA OR PIB OR desemprego", "dólar OR Ibovespa OR bolsa"],
    "Política": ["Congresso OR Senado OR Câmara", "STF OR Planalto OR ministro", "eleições OR partido OR governador"],
    "Fintechs": ["fintech OR fintechs", "Nubank OR Pix OR \"banco digital\"", '"open finance" OR criptomoeda OR bitcoin'],
    "Futebol": ["Brasileirão OR Libertadores", "Corinthians OR Palmeiras OR Flamengo", "seleção brasileira futebol OR Copa do Brasil"],
    "Motores": ['"Fórmula 1" OR F1 OR Verstappen', "Stock Car OR MotoGP OR automobilismo", "carro lançamento OR moto lançamento"],
    "Jiu-Jitsu": ['"jiu-jitsu" OR "jiu jitsu"', "IBJJF OR CBJJ OR faixa preta", "UFC OR MMA OR luta"],
}
DATA = Path(__file__).resolve().parent.parent / "data" / "headlines.csv"
HEADERS = {"User-Agent": "classificador-manchetes/1.0 (+https://github.com/YagoXavier10)"}
FIELDS = ["title", "topic", "source", "published"]


def fetch(query):
    url = "https://news.google.com/rss/search?q=" + quote_plus(query + " when:30d") + "&hl=pt-BR&gl=BR&ceid=BR:pt-419"
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    for item in ET.fromstring(resp.content).findall("./channel/item"):
        title = item.findtext("title", "").strip()
        source = item.findtext("source", "").strip()
        if source and title.endswith(" - " + source):
            title = title[: -len(" - " + source)]
        yield title, source, item.findtext("pubDate", "")


def main():
    rows = {}
    if DATA.exists():
        with DATA.open(encoding="utf-8") as f:
            rows = {r["title"]: r for r in csv.DictReader(f)}
    before = len(rows)
    for topic, queries in TOPICS.items():
        for q in queries:
            try:
                for title, source, pub in fetch(q):
                    rows.setdefault(title, {"title": title, "topic": topic, "source": source, "published": pub})
            except requests.RequestException as e:
                print(f"falhou: {topic} / {q}: {e}")
            time.sleep(1)
    DATA.parent.mkdir(exist_ok=True)
    with DATA.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(sorted(rows.values(), key=lambda r: (r["topic"], r["title"])))
    print(f"{len(rows)} manchetes ({len(rows) - before} novas)")


if __name__ == "__main__":
    main()
