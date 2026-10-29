"""Funções compartilhadas: o mesmo pré-processamento é refeito em JavaScript no site (app.js)."""
import re
import unicodedata

STOP_WORDS = sorted(set("""
a ao aos as ate com como da das de dei do dos e ela ele eles em entre era essa esse esta este eu foi
for ha isso ja la mais mas me mesmo meu muito na nas nao no nos o os ou para pela pelo por qual quando
que se sem ser seu sua so sobre tambem tem um uma umas uns vai vao voce apos diz contra novo nova
""".split()))

TOKEN = re.compile(r"[a-z0-9]{2,}")


def normalize(text):
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def tokens(text):
    words = [w for w in TOKEN.findall(normalize(text)) if w not in STOP_WORDS]
    return words + [f"{a} {b}" for a, b in zip(words, words[1:])]
