// Search for ART hyperparameters that yield exactly 12 hubs on the synthetic_12_data
// dataset, where each hub maps cleanly to one of the 12 prototypes.

const fs = require('fs'), path = require('path');
const d = fs.readFileSync(path.join(__dirname, '..', 'public', 'synthetic_12_data.js'), 'utf-8');
const csv = d.match(/`([\s\S]+?)`/)[1].trim();
const lines = csv.split('\n'); const cols = lines[0].split(',').slice(1);
const rows = lines.slice(1).map(l => { const p = l.split(','); return { name: p[0], x: p.slice(1).map(v => parseFloat(v) || 0) }; });
const order = []; const map = {};
cols.forEach(c => { const k = c.split(':')[0]; if (!map[k]) { map[k] = []; order.push(k); } map[k].push(c); });
const ps = order.map(p => map[p].length);
const X = rows.map(r => r.x);
const CONCEPTS = ['pine','oak','tree','rose','daisy','flower','robin','canary','bird','sunfish','salmon','fish'];

// Per-pool normalized Euclidean ART (matches public/synthetic_12.html UnsupervisedModel).
class M {
    constructor(ps, v, lr, g, inh) { this.ps = ps; this.v = v; this.lr = lr; this.gain = g; this.inh = inh; this.t = []; this.sl = []; let s = 0; for (const z of ps) { this.sl.push([s, s + z]); s += z; } }
    rec(x) { this.t.push([...x]); return this.t.length - 1; }
    mask(x) { return this.sl.map(([s, e]) => { let n = 0; for (let i = s; i < e; i++) n += x[i] * x[i]; return n > 1e-8; }); }
    sims(x, y) { const r = []; for (const [s, e] of this.sl) { const n = e - s; let d2 = 0; for (let i = s; i < e; i++) { const d = x[i] - y[i]; d2 += d * d; } r.push(1 - d2 / n); } return r; }
    dr(x) { if (!this.t.length) return []; const mk = this.mask(x); return this.t.map(t => { const s = this.sims(x, t); let sum = 0, c = 0; for (let i = 0; i < s.length; i++) if (mk[i]) { sum += s[i]; c++; } return this.gain * (c > 0 ? sum / c : 0); }); }
    set(x) { const d = this.dr(x); if (!d.length) return [null]; let a = new Array(d.length).fill(0); for (let s = 0; s < 20; s++) { const tot = a.reduce((p, q) => p + q, 0); a = a.map((v, i) => Math.max(0, d[i] - this.inh * (tot - v))); } let w = 0, mx = a[0]; for (let i = 1; i < a.length; i++) if (a[i] > mx) { mx = a[i]; w = i; } return [w]; }
    step(x) { if (!this.t.length) { this.rec(x); return; } const [w] = this.set(x); const y = [...this.t[w]]; const s = this.sims(x, y), mk = this.mask(x); if (s.every((v, i) => !mk[i] || v >= this.v)) { const o = this.t[w]; this.t[w] = o.map((q, i) => (1 - this.lr) * q + this.lr * x[i]); } else this.rec(x); }
}

function score(v, lr, gain, inh, passes) {
    const m = new M(ps, v, lr, gain, inh);
    for (let p = 0; p < passes; p++) for (const x of X) m.step(x);
    const finals = X.map(x => m.set(x)[0]);
    const hubConc = {}, conHub = {};
    for (let i = 0; i < rows.length; i++) {
        const c = rows[i].name.replace(/\d+$/, '');
        (hubConc[finals[i]] = hubConc[finals[i]] || {})[c] = (hubConc[finals[i]][c] || 0) + 1;
        (conHub[c] = conHub[c] || {})[finals[i]] = (conHub[c][finals[i]] || 0) + 1;
    }
    const distinct = new Set(finals).size;
    return { v, lr, gain, inh, passes, hubs: m.t.length, distinct, conHub, hubConc };
}

function scoreOneHubPerConcept(r) {
    let perfect = 0;
    for (const c of CONCEPTS) {
        if (!r.conHub[c]) continue;
        const top = Object.entries(r.conHub[c]).sort((a, b) => b[1] - a[1])[0];
        if (!top) continue;
        const [hub, count] = top;
        const hubIsPure = Object.keys(r.hubConc[hub]).length === 1;
        if (hubIsPure && count === 20) perfect++;
    }
    return perfect;
}

const results = [];
for (const v of [0.93, 0.95, 0.96, 0.97, 0.975, 0.977, 0.98, 0.985, 0.99])
for (const lr of [0.02, 0.05, 0.08, 0.1, 0.15, 0.2])
for (const gain of [1.0])
for (const inh of [0, 0.1])
for (const passes of [1, 2, 3]) {
    results.push(score(v, lr, gain, inh, passes));
}

console.log(`Searched ${results.length} configs`);

const exact12 = results.filter(r => r.distinct === 12);
console.log(`Configs with exactly 12 distinct hubs: ${exact12.length}\n`);

const ranked = exact12.map(r => ({ ...r, perfect: scoreOneHubPerConcept(r) }))
    .sort((a, b) => b.perfect - a.perfect || a.hubs - b.hubs);

console.log('Top 10 "exactly 12 hubs" configs by perfect-concept count:');
ranked.slice(0, 10).forEach((r, i) => {
    console.log(`\n#${i + 1}  ${r.perfect}/12 perfect · v=${r.v} lr=${r.lr} g=${r.gain} inh=${r.inh} p=${r.passes}  (${r.hubs} hubs)`);
    for (const c of CONCEPTS) {
        const items = Object.entries(r.conHub[c] || {}).sort((a, b) => b[1] - a[1])
            .map(([h, n]) => `H${h}${Object.keys(r.hubConc[h]).length > 1 ? '*' : ''}×${n}`).join(' ');
        console.log(`  ${c.padEnd(8)} ${items}`);
    }
});

if (exact12.length === 0) {
    console.log('No exact-12 config found. Showing closest by distinct hub count:');
    const sorted = results.sort((a, b) => Math.abs(a.distinct - 12) - Math.abs(b.distinct - 12));
    sorted.slice(0, 5).forEach((r, i) => {
        console.log(`\n#${i + 1}  ${r.distinct} distinct hubs · v=${r.v} lr=${r.lr} g=${r.gain} inh=${r.inh} p=${r.passes}`);
        for (const c of CONCEPTS) {
            const items = Object.entries(r.conHub[c] || {}).sort((a, b) => b[1] - a[1])
                .map(([h, n]) => `H${h}${Object.keys(r.hubConc[h]).length > 1 ? '*' : ''}×${n}`).join(' ');
            console.log(`  ${c.padEnd(8)} ${items}`);
        }
    });
}
