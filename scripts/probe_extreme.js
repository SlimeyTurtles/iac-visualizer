const fs = require('fs');
const path = require('path');
const dataJs = fs.readFileSync(path.join(__dirname, '..', 'public', 'sensory_data.js'), 'utf-8');
const csvText = dataJs.match(/`([\s\S]+?)`/)[1];
const lines = csvText.trim().split('\n');
const cols = lines[0].split(',').slice(1);
const rows = lines.slice(1).map(l => { const p = l.split(','); return { name: p[0], x: p.slice(1).map(v => parseFloat(v) || 0) }; });
const poolOrder = []; const poolMap = {};
cols.forEach(c => { const k = c.split(':')[0]; if (!poolMap[k]) { poolMap[k] = []; poolOrder.push(k); } poolMap[k].push(c); });
const pool_sizes = poolOrder.map(p => poolMap[p].length);
const X = rows.map(r => r.x);

// Per-pool normalized Euclidean similarity ART (matches public/sensory_dropout.html).
class M {
    constructor(ps, v, lr, g, inh) { this.ps = ps; this.v = v; this.lr = lr; this.gain = g; this.inh = inh; this.tmpls = []; this.sl = []; let s = 0; for (const z of ps) { this.sl.push([s, s + z]); s += z; } }
    recruit(x) { this.tmpls.push([...x]); return this.tmpls.length - 1; }
    mask(x) { return this.sl.map(([s, e]) => { let n = 0; for (let i = s; i < e; i++) n += x[i] * x[i]; return n > 1e-8; }); }
    poolSims(x, y) { const r = []; for (const [s, e] of this.sl) { const n = e - s; let d2 = 0; for (let i = s; i < e; i++) { const d = x[i] - y[i]; d2 += d * d; } r.push(1 - d2 / n); } return r; }
    drive(x) { if (!this.tmpls.length) return []; const mk = this.mask(x); return this.tmpls.map(t => { const s = this.poolSims(x, t); let sum = 0, c = 0; for (let i = 0; i < s.length; i++) if (mk[i]) { sum += s[i]; c++; } return this.gain * (c > 0 ? sum / c : 0); }); }
    settle(x) { const d = this.drive(x); if (!d.length) return [null]; let a = new Array(d.length).fill(0); for (let s = 0; s < 20; s++) { const tot = a.reduce((p, q) => p + q, 0); a = a.map((v, i) => Math.max(0, d[i] - this.inh * (tot - v))); } let w = 0, mx = a[0]; for (let i = 1; i < a.length; i++) if (a[i] > mx) { mx = a[i]; w = i; } return [w]; }
    step(x) { if (!this.tmpls.length) { this.recruit(x); return; } const [w] = this.settle(x); const y = [...this.tmpls[w]]; const sims = this.poolSims(x, y), mk = this.mask(x); if (sims.every((s, i) => !mk[i] || s >= this.v)) { const o = this.tmpls[w]; this.tmpls[w] = o.map((q, i) => (1 - this.lr) * q + this.lr * x[i]); } else this.recruit(x); }
}

function probe(v, lr, gain, inh, passes) {
    const m = new M(pool_sizes, v, lr, gain, inh);
    for (let p = 0; p < passes; p++) for (const x of X) m.step(x);
    const final = X.map(x => m.settle(x)[0]);
    const hubConcepts = {};
    const conceptHubs = {};
    for (let i = 0; i < rows.length; i++) {
        const c = rows[i].name.replace(/\d+$/, '');
        (hubConcepts[final[i]] = hubConcepts[final[i]] || {})[c] = (hubConcepts[final[i]][c] || 0) + 1;
        (conceptHubs[c] = conceptHubs[c] || {})[final[i]] = (conceptHubs[c][final[i]] || 0) + 1;
    }
    const concepts = ['pine', 'oak', 'rose', 'daisy', 'robin', 'canary', 'sunfish', 'salmon'];
    let resolved = 0;
    for (const c of concepts) {
        let pure = 0;
        for (const [h, n] of Object.entries(conceptHubs[c] || {})) {
            if (Object.keys(hubConcepts[h]).length === 1) pure += n;
        }
        if (pure >= 18) resolved++;
    }
    return { v, lr, gain, inh, passes, resolved, num_hubs: m.tmpls.length, conceptHubs, hubConcepts };
}

const tests = [
    [0.985, 0.05, 1, 0, 5],
    [0.988, 0.05, 1, 0, 5],
    [0.99, 0.02, 1, 0, 5],
    [0.99, 0.05, 1, 0, 3],
    [0.99, 0.05, 1, 0, 5],
    [0.99, 0.05, 1, 0, 8],
    [0.992, 0.05, 1, 0, 3],
    [0.995, 0.05, 1, 0, 3],
    [0.997, 0.05, 1, 0, 3],
];
for (const t of tests) {
    const r = probe(...t);
    console.log(`v=${r.v} lr=${r.lr} g=${r.gain} inh=${r.inh} p=${r.passes}: resolved=${r.resolved}/8, ${r.num_hubs} hubs`);
    for (const c of ['sunfish', 'salmon']) {
        const dist = Object.entries(r.conceptHubs[c] || {}).sort((a, b) => b[1] - a[1]).slice(0, 4)
            .map(([h, n]) => `H${h}${Object.keys(r.hubConcepts[h]).length > 1 ? '*' : ''}×${n}`).join(' ');
        console.log(`   ${c.padEnd(8)} ${dist}`);
    }
}
