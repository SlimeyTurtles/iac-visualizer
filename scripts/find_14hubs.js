// Grid search over (vigilance, learning_rate, passes, training_order) for the
// synthetic_12_clean dataset (14 prototypes). Reports configs where the model
// learns exactly 14 hubs AND each hub maps to exactly one concept (one species
// or generic) with all 20 of that concept's samples in it ("perfect").

const fs = require('fs');
const path = require('path');

const dataJs = fs.readFileSync(path.join(__dirname, '..', 'public', 'synthetic_12_clean_data.js'), 'utf-8');
const csv = dataJs.match(/`([\s\S]+?)`/)[1].trim();
const lines = csv.split('\n');
const cols = lines[0].split(',').slice(1);
const rows = lines.slice(1).map(l => {
    const p = l.split(',');
    return { name: p[0], x: p.slice(1).map(v => parseFloat(v) || 0) };
});

const poolOrder = [];
const poolCols = {};
cols.forEach(c => {
    const k = c.split(':')[0];
    if (!poolCols[k]) { poolCols[k] = []; poolOrder.push(k); }
    poolCols[k].push(c);
});
const pool_sizes = poolOrder.map(p => poolCols[p].length);
const X = rows.map(r => r.x);

const CONCEPTS = [
    'pine', 'oak', 'tree',
    'rose', 'daisy', 'flower',
    'plant',
    'robin', 'canary', 'bird',
    'sunfish', 'salmon', 'fish',
    'animal',
];
const GENERIC_ROOTS = new Set(['tree', 'flower', 'bird', 'fish', 'plant', 'animal']);
const rootOf = n => n.replace(/\d+$/, '');

// ---- Model (matches public/synthetic_12_clean.html UnsupervisedModel) ----
class M {
    constructor(ps, v, lr, gain, inh) {
        this.ps = ps; this.v = v; this.lr = lr; this.gain = gain; this.inh = inh;
        this.t = []; this.sl = []; let s = 0;
        for (const z of ps) { this.sl.push([s, s + z]); s += z; }
    }
    rec(x) { this.t.push([...x]); return this.t.length - 1; }
    mask(x) {
        return this.sl.map(([s, e]) => {
            let n = 0; for (let i = s; i < e; i++) n += x[i] * x[i];
            return n > 1e-8;
        });
    }
    sims(x, y) {
        const r = [];
        for (const [s, e] of this.sl) {
            const n = e - s; let d2 = 0;
            for (let i = s; i < e; i++) { const d = x[i] - y[i]; d2 += d * d; }
            r.push(1 - d2 / n);
        }
        return r;
    }
    drive(x) {
        if (!this.t.length) return [];
        const mk = this.mask(x);
        return this.t.map(t => {
            const s = this.sims(x, t);
            let sum = 0, c = 0;
            for (let i = 0; i < s.length; i++) if (mk[i]) { sum += s[i]; c++; }
            return this.gain * (c > 0 ? sum / c : 0);
        });
    }
    settle(x) {
        const d = this.drive(x);
        if (!d.length) return [null];
        let a = new Array(d.length).fill(0);
        for (let s = 0; s < 20; s++) {
            const tot = a.reduce((p, q) => p + q, 0);
            a = a.map((v, i) => Math.max(0, d[i] - this.inh * (tot - v)));
        }
        let w = 0, mx = a[0];
        for (let i = 1; i < a.length; i++) if (a[i] > mx) { mx = a[i]; w = i; }
        return [w];
    }
    step(x) {
        if (!this.t.length) { this.rec(x); return; }
        const [w] = this.settle(x);
        const y = [...this.t[w]];
        const s = this.sims(x, y), mk = this.mask(x);
        if (s.every((v, i) => !mk[i] || v >= this.v)) {
            const o = this.t[w];
            this.t[w] = o.map((q, i) => (1 - this.lr) * q + this.lr * x[i]);
        } else this.rec(x);
    }
}

// ---- Training orders (matches buildTrainingOrder in the page) ----
function mulberry32(seed) {
    return function () {
        seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
        let t = seed;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

function buildOrder(mode, episodes) {
    const n = episodes.length;
    const idx = Array.from({ length: n }, (_, i) => i);
    if (mode === 'high-to-low') {
        const gen = idx.filter(i => GENERIC_ROOTS.has(rootOf(episodes[i].name)));
        const sp = idx.filter(i => !GENERIC_ROOTS.has(rootOf(episodes[i].name)));
        return [...gen, ...sp];
    }
    if (mode === 'low-to-high') {
        const sp = idx.filter(i => !GENERIC_ROOTS.has(rootOf(episodes[i].name)));
        const gen = idx.filter(i => GENERIC_ROOTS.has(rootOf(episodes[i].name)));
        return [...sp, ...gen];
    }
    if (mode === 'random') {
        const rng = mulberry32(42);
        const shuf = [...idx];
        for (let i = shuf.length - 1; i > 0; i--) {
            const j = Math.floor(rng() * (i + 1));
            [shuf[i], shuf[j]] = [shuf[j], shuf[i]];
        }
        return shuf;
    }
    return idx;
}

const episodes = rows.map(r => ({ name: r.name }));

function score(v, lr, gain, inh, passes, orderMode) {
    const m = new M(pool_sizes, v, lr, gain, inh);
    const order = buildOrder(orderMode, episodes);
    for (let p = 0; p < passes; p++) for (const i of order) m.step(X[i]);
    const finals = X.map(x => m.settle(x)[0]);
    const hubConc = {}, conHub = {};
    for (let i = 0; i < rows.length; i++) {
        const c = rootOf(rows[i].name);
        (hubConc[finals[i]] = hubConc[finals[i]] || {})[c] = (hubConc[finals[i]][c] || 0) + 1;
        (conHub[c] = conHub[c] || {})[finals[i]] = (conHub[c][finals[i]] || 0) + 1;
    }
    const distinct = new Set(finals).size;
    let perfect = 0;
    for (const c of CONCEPTS) {
        if (!conHub[c]) continue;
        const top = Object.entries(conHub[c]).sort((a, b) => b[1] - a[1])[0];
        const [hub, count] = top;
        const hubIsPure = Object.keys(hubConc[hub]).length === 1;
        if (hubIsPure && count === 20) perfect++;
    }
    return { v, lr, gain, inh, passes, orderMode, hubs: m.t.length, distinct, perfect, hubConc, conHub };
}

// ---- Grid ----
const V_GRID  = [0.900, 0.920, 0.930, 0.935, 0.940, 0.945, 0.950, 0.955, 0.960, 0.965, 0.970, 0.975, 0.978, 0.980, 0.982, 0.985, 0.988, 0.990, 0.992, 0.995, 0.997, 0.999];
const LR_GRID = [0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 0.90];
const P_GRID  = [1, 2, 3, 4, 5, 7, 10];
const ORDERS  = ['default', 'random', 'high-to-low', 'low-to-high'];
const GAIN = 1.0;
const INH  = 0.0;

const results = [];
for (const v of V_GRID)
for (const lr of LR_GRID)
for (const passes of P_GRID)
for (const order of ORDERS) {
    results.push(score(v, lr, GAIN, INH, passes, order));
}
console.log(`Searched ${results.length} configs (gain=${GAIN}, inhibition=${INH})\n`);

const perfect14 = results.filter(r => r.distinct === 14 && r.perfect === 14);
console.log(`Configs with 14 distinct hubs AND 14/14 perfect concept assignments: ${perfect14.length}\n`);

// ---- Per-order summary table ----
console.log('=== Working configs grouped by training order ===');
for (const order of ORDERS) {
    const matches = perfect14.filter(r => r.orderMode === order);
    console.log(`\n[order=${order}]  ${matches.length} working configs`);
    if (!matches.length) { console.log('  (none)'); continue; }
    const byVigilance = new Map();
    for (const r of matches) {
        if (!byVigilance.has(r.v)) byVigilance.set(r.v, []);
        byVigilance.get(r.v).push(r);
    }
    const header = '  vigilance  | learning rates that work × passes-needed';
    console.log(header);
    console.log('  ' + '-'.repeat(header.length));
    [...byVigilance.keys()].sort((a, b) => a - b).forEach(v => {
        const rs = byVigilance.get(v);
        const byLR = new Map();
        for (const r of rs) {
            if (!byLR.has(r.lr)) byLR.set(r.lr, []);
            byLR.get(r.lr).push(r.passes);
        }
        const parts = [...byLR.entries()].sort((a, b) => a[0] - b[0])
            .map(([lr, passes]) => `lr=${lr.toFixed(2)} (passes: ${passes.sort((a, b) => a - b).join(',')})`);
        console.log(`  ${v.toFixed(3)}    | ${parts.join('  ')}`);
    });
}

// ---- Range table: parameter envelope per training order ----
console.log('\n\n=== Parameter envelope per training order (range of values that produced 14/14) ===');
console.log('  order         | vigilance range | learning-rate range | passes range | n configs');
console.log('  ' + '-'.repeat(95));
for (const order of ORDERS) {
    const matches = perfect14.filter(r => r.orderMode === order);
    if (!matches.length) {
        console.log(`  ${order.padEnd(13)} | (no working configs)`);
        continue;
    }
    const vs = [...new Set(matches.map(r => r.v))].sort((a, b) => a - b);
    const lrs = [...new Set(matches.map(r => r.lr))].sort((a, b) => a - b);
    const ps = [...new Set(matches.map(r => r.passes))].sort((a, b) => a - b);
    console.log(
        `  ${order.padEnd(13)} | ` +
        `${vs[0].toFixed(3)} – ${vs[vs.length - 1].toFixed(3)}     | ` +
        `${lrs[0].toFixed(2)} – ${lrs[lrs.length - 1].toFixed(2)}           | ` +
        `${ps[0]} – ${ps[ps.length - 1]}          | ` +
        `${matches.length}`
    );
}

// ---- "Near miss" diagnostic: distinct=14 but not perfect (i.e. one concept stole another's hub) ----
const near14 = results.filter(r => r.distinct === 14 && r.perfect < 14);
console.log(`\n\nNear-miss configs (14 distinct hubs but not all 14 concepts cleanly assigned): ${near14.length}`);

// ---- Concept-failure breakdown across all "near miss + perfect" 14-hub configs ----
const all14 = results.filter(r => r.distinct === 14);
if (all14.length) {
    const concFails = {};
    for (const r of all14) {
        for (const c of CONCEPTS) {
            if (!r.conHub[c]) continue;
            const top = Object.entries(r.conHub[c]).sort((a, b) => b[1] - a[1])[0];
            const pure = Object.keys(r.hubConc[top[0]]).length === 1;
            if (!(pure && top[1] === 20)) concFails[c] = (concFails[c] || 0) + 1;
        }
    }
    if (Object.keys(concFails).length) {
        console.log('\nWhich concepts most often fail (among configs that hit 14 hubs but missed perfect):');
        Object.entries(concFails).sort((a, b) => b[1] - a[1])
            .forEach(([c, n]) => console.log(`  ${c.padEnd(8)} failed in ${n}/${all14.length} 14-hub configs`));
    }
}
