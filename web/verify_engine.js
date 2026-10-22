// node web/verify_engine.js : the JS engine must give the same answers as src/cbr.py
const fs = require("fs"), path = require("path");
const { Engine } = require("./engine.js");
const D = JSON.parse(fs.readFileSync(path.join(__dirname, "data.json")));
const R = JSON.parse(fs.readFileSync(path.join(__dirname, "reference.json")));
const eng = new Engine(D.config, D.cases);
const n = R.preds.length;
const client = (r) => ({ num: D.config.num.map((_, j) => R.clients.num[j][r]), cat: D.config.cat.map((_, j) => R.clients.cat[j][r]) });
let ok = true;

let same = 0, maxDiff = 0, exact = 0;
for (let r = 0; r < n; r++) {
  const p = eng.predict(eng.encode(client(r)));
  if (p.pred === R.preds[r]) same++;
  const d = Math.abs(p.score - R.scores[r]);
  maxDiff = Math.max(maxDiff, d); if (d < 1e-9) exact++;
}
// Python sums distances with a matrix product, JS with a loop: two neighbours tied at position k can be
// rounded differently and swapped, moving the score by about 1/k. Predictions must still be identical.
console.log(`static: same prediction ${same}/${n}, identical score ${exact}/${n}, largest difference ${maxDiff.toFixed(4)}`);
ok = ok && same === n && exact >= 0.98 * n && maxDiff < 0.02;

for (const [name, rel] of [["retain", false], ["revise_reliability", true]]) {
  eng.reset(); eng.useReliability = rel;
  let s = 0;
  for (let r = 0; r < n; r++) {
    const c = client(r), p = eng.predict(eng.encode(c)), truth = R.clients.y[r];
    if (p.pred === R.online[name].preds[r]) s++;
    eng.revise(p.nb.idx, p.pred, truth); eng.retain(c, truth);
  }
  let relSum = 0; for (let i = 0; i < eng.n; i++) relSum += eng.rel[i];
  const good = s === n && eng.n === R.online[name].n && Math.abs(relSum - R.online[name].reliability_sum) < 1e-6;
  console.log(`online (${name}): same prediction ${s}/${n}, case base ${eng.n}, reliability sum ${relSum.toFixed(4)} vs ${R.online[name].reliability_sum.toFixed(4)} ${good ? "OK" : "MISMATCH"}`);
  ok = ok && good;
}
console.log(ok ? "ALL CHECKS PASSED" : "CHECKS FAILED");
process.exit(ok ? 0 : 1);
