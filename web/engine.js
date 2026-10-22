/* The CBR engine of src/cbr.py, in JavaScript, for the web demo.
   Checked against the Python engine by web/verify_engine.js. */
(function (root) {
  "use strict";

  function Engine(cfg, cases) {
    this.cfg = cfg; this.NUM = cfg.num; this.CAT = cfg.cat;
    this.n0 = cases.y.length; this.cap = this.n0 + 5000; this.initial = cases;
    this.X = this.NUM.map(() => new Float64Array(this.cap));     // scaled numeric
    this.raw = this.NUM.map(() => new Float64Array(this.cap));   // original numeric
    this.C = this.CAT.map(() => new Int16Array(this.cap));
    this.y = new Uint8Array(this.cap); this.rel = new Float64Array(this.cap);
    this.rowId = new Int32Array(this.cap); this.added = new Uint8Array(this.cap);
    this.k = cfg.best.k; this.vote = cfg.best.vote; this.useReliability = false;
    this.decay = cfg.decay; this.minRel = cfg.min_reliability;
    this.setScheme(cfg.best.weights);
    this.reset();
  }

  /* same as preprocessing.py: 1st-99th percentile of the training clients, clipped to [0, 1] */
  Engine.prototype.scale = function (j, v) {
    const c = this.NUM[j], lo = this.cfg.low[c], hi = this.cfg.high[c];
    const s = (v - lo) / ((hi - lo) || 1);
    return s < 0 ? 0 : s > 1 ? 1 : s;
  };

  Engine.prototype.setScheme = function (name) {
    const w = this.cfg.weights[name];
    this.scheme = name;
    this.wNum = this.NUM.map((c) => w[c]); this.wCat = this.CAT.map((c) => w[c]);
    this.wSum = this.wNum.concat(this.wCat).reduce((a, b) => a + b, 0);
  };

  Engine.prototype.reset = function () {
    const cs = this.initial; this.n = this.n0; this.nYes = 0;
    for (let j = 0; j < this.NUM.length; j++) for (let i = 0; i < this.n0; i++) {
      this.raw[j][i] = cs.num[j][i]; this.X[j][i] = this.scale(j, cs.num[j][i]);
    }
    for (let j = 0; j < this.CAT.length; j++) this.C[j].set(cs.cat[j]);
    for (let i = 0; i < this.n0; i++) {
      this.y[i] = cs.y[i]; this.rel[i] = 1; this.rowId[i] = cs.row_id[i]; this.added[i] = 0; this.nYes += cs.y[i];
    }
  };

  Engine.prototype.baseRate = function () { return this.nYes / this.n; };

  /* client = {num: [raw values], cat: [codes]} */
  Engine.prototype.encode = function (client) {
    return { x: client.num.map((v, j) => this.scale(j, v)), c: client.cat.slice() };
  };

  /* 1. RETRIEVE: k smallest distances, best first; ties -> earlier case first */
  Engine.prototype.neighbours = function (q, k) {
    k = Math.min(k || this.k, this.n);
    const n = this.n, d = new Float64Array(n);
    for (let j = 0; j < this.NUM.length; j++) {
      const w = this.wNum[j]; if (!w) continue;
      const col = this.X[j], a = q.x[j];
      for (let i = 0; i < n; i++) d[i] += w * Math.abs(col[i] - a);
    }
    for (let j = 0; j < this.CAT.length; j++) {
      const w = this.wCat[j]; if (!w) continue;
      const col = this.C[j], a = q.c[j];
      for (let i = 0; i < n; i++) if (col[i] !== a) d[i] += w;
    }
    const idx = [], dist = [];
    for (let i = 0; i < n; i++) {
      const v = d[i] / this.wSum;
      if (idx.length === k && v >= dist[k - 1]) continue;
      let p = idx.length;
      while (p > 0 && dist[p - 1] > v) p--;
      idx.splice(p, 0, i); dist.splice(p, 0, v);
      if (idx.length > k) { idx.pop(); dist.pop(); }
    }
    return { idx: idx, sim: dist.map((v) => 1 - v) };
  };

  /* 2. REUSE: majority / balanced / similarity-weighted balanced (see src/cbr.py vote()) */
  Engine.prototype.decide = function (nb, rule) {
    rule = rule || this.vote;
    const k = nb.idx.length, base = this.baseRate();
    let yes = 0;
    for (const i of nb.idx) yes += this.y[i];
    if (rule === "majority") {
      const share = yes / k;
      let pred = share > 0.5 ? 1 : 0;
      if (Math.abs(share - 0.5) < 1e-12) pred = this.y[nb.idx[0]];
      return { pred: pred, score: share, threshold: 0.5, nYes: yes };
    }
    let num = 0, den = 0;
    for (let t = 0; t < k; t++) {
      const i = nb.idx[t], w = (rule === "weighted_balanced" ? Math.max(nb.sim[t], 1e-6) : 1) * this.rel[i];
      num += w * this.y[i]; den += w;
    }
    const score = num / den;
    return { pred: score >= base ? 1 : 0, score: score, threshold: base, nYes: yes };
  };

  Engine.prototype.predict = function (q) {
    const nb = this.neighbours(q), d = this.decide(nb);
    return { nb: nb, score: d.score, pred: d.pred, threshold: d.threshold, nYes: d.nYes };
  };

  /* 3. REVISE: the solution kept is the real outcome; optionally re-weight the neighbours */
  Engine.prototype.revise = function (idx, pred, truth) {
    const changed = [], wrong = pred !== truth;
    if (this.useReliability) {
      if (wrong) { for (const i of idx) if (this.y[i] === pred) { this.rel[i] = Math.max(this.minRel, this.rel[i] * this.decay); changed.push(i); } }
      else { for (const i of idx) if (this.y[i] === truth) { const b = this.rel[i]; this.rel[i] = Math.min(1, this.rel[i] / this.decay); if (this.rel[i] !== b) changed.push(i); } }
    }
    return { wrong: wrong, changed: changed };
  };

  /* 4. RETAIN: add the client with the REAL outcome */
  Engine.prototype.retain = function (client, truth) {
    if (this.n >= this.cap) throw new Error("Case base full: reset the demo");
    const i = this.n;
    for (let j = 0; j < this.NUM.length; j++) { this.raw[j][i] = client.num[j]; this.X[j][i] = this.scale(j, client.num[j]); }
    for (let j = 0; j < this.CAT.length; j++) this.C[j][i] = client.cat[j];
    this.y[i] = truth; this.rel[i] = 1; this.rowId[i] = client.row_id == null ? -1 : client.row_id; this.added[i] = 1;
    this.nYes += truth; this.n += 1;
    return this.n;
  };

  if (typeof module !== "undefined" && module.exports) module.exports = { Engine: Engine }; else root.FriendCBR = { Engine: Engine };
})(typeof window !== "undefined" ? window : this);
