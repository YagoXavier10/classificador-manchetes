// Refaz em JavaScript a mesma conta do scikit-learn: tokens -> TF-IDF -> regressão logística -> softmax.
const EXAMPLES = [
  "Banco Central mantém a Selic e dólar fecha em queda",
  "OpenAI lança novo modelo de IA capaz de programar sozinho",
  "Corinthians vence clássico e sobe na tabela do Brasileirão",
  "Brasileiro conquista ouro no Mundial de Jiu-Jitsu da IBJJF",
  "Verstappen larga na pole no GP de São Paulo de Fórmula 1",
  "Pix bate recorde de transações em um único dia",
];

let model, index, stop;

const normalize = s => s.toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "");

function tokens(text) {
  const words = (normalize(text).match(/[a-z0-9]{2,}/g) || []).filter(w => !stop.has(w));
  const grams = [];
  for (let i = 0; i < words.length - 1; i++) grams.push(words[i] + " " + words[i + 1]);
  return words.concat(grams);
}

function predict(text) {
  const tf = new Map();
  for (const t of tokens(text)) {
    const j = index.get(t);
    if (j !== undefined) tf.set(j, (tf.get(j) || 0) + 1);
  }
  const feats = [...tf].map(([j, c]) => [j, (1 + Math.log(c)) * model.idf[j]]);
  const norm = Math.hypot(...feats.map(f => f[1])) || 1;
  const x = feats.map(([j, v]) => [j, v / norm]);
  const scores = model.classes.map((_, k) => model.intercept[k] + x.reduce((s, [j, v]) => s + v * model.coef[k][j], 0));
  const max = Math.max(...scores);
  const exp = scores.map(s => Math.exp(s - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  const probs = model.classes.map((c, k) => ({ c, k, p: exp[k] / sum })).sort((a, b) => b.p - a.p);
  const best = probs[0].k;
  const why = x.map(([j, v]) => ({ w: model.vocab[j], s: v * model.coef[best][j] }))
    .filter(e => e.s > 0).sort((a, b) => b.s - a.s).slice(0, 4).map(e => e.w);
  return { probs, why, known: x.length };
}

const pct = p => (p * 100).toFixed(p >= .995 || p < .01 ? 0 : 1) + "%";

function render() {
  const text = document.getElementById("headline").value.trim();
  const result = document.getElementById("result");
  const whyEl = document.getElementById("why");
  if (!text) { result.innerHTML = '<p class="muted">Escreva uma manchete ou escolha um exemplo acima.</p>'; whyEl.innerHTML = ""; return; }
  const { probs, why, known } = predict(text);
  if (!known) {
    result.innerHTML = '<p class="muted">Nenhuma palavra dessa manchete aparece no vocabulário do modelo. Tente uma frase mais parecida com uma notícia.</p>';
    whyEl.innerHTML = ""; return;
  }
  result.innerHTML = `<div class="top"><span class="topic">${probs[0].c}</span><span class="conf">${pct(probs[0].p)} de confiança</span></div>` +
    probs.map((r, i) => `<div class="row${i ? "" : " best"}"><span>${r.c}</span><div class="track"><div class="fill" data-w="${(r.p * 100).toFixed(1)}"></div></div><span class="pct">${pct(r.p)}</span></div>`).join("");
  requestAnimationFrame(() => result.querySelectorAll(".fill").forEach(f => { f.style.width = f.dataset.w + "%"; }));
  whyEl.innerHTML = why.length ? "Palavras que mais pesaram: " + why.map(w => `<mark>${w}</mark>`).join(" ") : "";
}

function renderMetrics(m) {
  const date = new Date(m.trained_at).toLocaleDateString("pt-BR");
  document.getElementById("trained").textContent = "modelo treinado em " + date;
  document.getElementById("stats").innerHTML = [
    [(m.accuracy * 100).toFixed(1) + "%", "de acerto no teste"],
    [m.n_headlines.toLocaleString("pt-BR"), "manchetes no treino"],
    [m.labels.length, "temas"],
    [m.vocab_size.toLocaleString("pt-BR"), "palavras no vocabulário"],
  ].map(([b, s]) => `<div class="stat"><b>${b}</b><span>${s}</span></div>`).join("");

  const f1 = Object.entries(m.per_topic).sort((a, b) => b[1].f1 - a[1].f1);
  document.getElementById("f1").innerHTML = f1.map(([c, v]) =>
    `<div class="row"><span>${c}</span><div class="track"><div class="fill" data-w="${v.f1 * 100}"></div></div><span class="pct">${Math.round(v.f1 * 100)}%</span></div>`).join("");

  document.getElementById("words").innerHTML = Object.entries(m.top_words).map(([c, ws]) =>
    `<div class="wordrow"><b>${c}</b>${ws.slice(0, 6).map(w => `<span class="word">${w}</span>`).join("")}</div>`).join("");

  const max = Math.max(...m.confusion.flat());
  const short = c => c.replace("Inteligência Artificial", "IA").replace("Ciência de Dados", "Dados");
  document.getElementById("cm").innerHTML =
    `<tr><th></th>${m.labels.map(l => `<th class="col">${short(l)}</th>`).join("")}</tr>` +
    m.confusion.map((row, i) => `<tr><th>${short(m.labels[i])}</th>${row.map((v, j) => {
      const a = v / max;
      const bg = i === j ? `rgba(110,231,249,${.12 + a * .7})` : v ? `rgba(255,143,177,${.15 + a * 2})` : "rgba(255,255,255,.03)";
      return `<td style="background:${bg}">${v || ""}</td>`;
    }).join("")}</tr>`).join("");

  setTimeout(() => document.querySelectorAll("#f1 .fill").forEach(f => { f.style.width = f.dataset.w + "%"; }), 300);
}

async function init() {
  const [mdl, metrics] = await Promise.all([fetch("model.json").then(r => r.json()), fetch("metrics.json").then(r => r.json())]);
  model = mdl;
  index = new Map(model.vocab.map((w, i) => [w, i]));
  stop = new Set(model.stop_words);
  renderMetrics(metrics);

  const box = document.getElementById("headline");
  document.getElementById("examples").innerHTML = EXAMPLES.map(e => `<button class="chip" type="button">${e}</button>`).join("");
  document.querySelectorAll(".chip").forEach(b => b.addEventListener("click", () => { box.value = b.textContent; render(); }));
  box.addEventListener("input", render);
  box.value = EXAMPLES[0];
  render();
}

init().catch(() => {
  document.getElementById("result").innerHTML = '<p class="muted">Não consegui carregar o modelo. Abra o site por um servidor (veja o README).</p>';
});

// Fundo animado: uma rede de neurônios ligados que se mexe devagar.
(function network() {
  const cv = document.getElementById("net");
  if (!cv || matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const ctx = cv.getContext("2d");
  let w, h, pts;
  function size() {
    w = cv.width = innerWidth * devicePixelRatio;
    h = cv.height = innerHeight * devicePixelRatio;
    const n = Math.min(70, Math.round(innerWidth / 18));
    pts = Array.from({ length: n }, () => ({ x: Math.random() * w, y: Math.random() * h, vx: (Math.random() - .5) * .35, vy: (Math.random() - .5) * .35 }));
  }
  function frame() {
    ctx.clearRect(0, 0, w, h);
    const R = 150 * devicePixelRatio;
    for (const p of pts) {
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0 || p.x > w) p.vx *= -1;
      if (p.y < 0 || p.y > h) p.vy *= -1;
    }
    for (let i = 0; i < pts.length; i++) for (let j = i + 1; j < pts.length; j++) {
      const d = Math.hypot(pts[i].x - pts[j].x, pts[i].y - pts[j].y);
      if (d < R) {
        ctx.strokeStyle = `rgba(167,139,250,${(1 - d / R) * .35})`;
        ctx.lineWidth = devicePixelRatio;
        ctx.beginPath(); ctx.moveTo(pts[i].x, pts[i].y); ctx.lineTo(pts[j].x, pts[j].y); ctx.stroke();
      }
    }
    ctx.fillStyle = "rgba(110,231,249,.8)";
    for (const p of pts) { ctx.beginPath(); ctx.arc(p.x, p.y, 2 * devicePixelRatio, 0, 7); ctx.fill(); }
    requestAnimationFrame(frame);
  }
  addEventListener("resize", size);
  size(); frame();
})();
