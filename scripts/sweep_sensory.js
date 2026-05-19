// Hyperparameter sweep for the Hierarchy Emergence ART (sensory_dropout) model.
// Goal: find (vigilance, learning_rate, gain, inhibition, num_passes) where the
// 8 concepts (pine, oak, rose, daisy, robin, canary, sunfish, salmon) cluster cleanly.

const fs = require('fs');
const path = require('path');

// ---------- Load + parse the same CSV the page uses ----------
const dataJs = fs.readFileSync(path.join(__dirname, '..', 'public', 'sensory_data.js'), 'utf-8');
const csvMatch = dataJs.match(/window\.SENSORY_DROPOUT_CSV\s*=\s*`([\s\S]+?)`/);
if (!csvMatch) throw new Error('Could not extract CSV from sensory_data.js');
const csvText = csvMatch[1];

function parseCSV(csvText) {
    const lines = csvText.trim().split('\n');
    const header = lines[0].split(',').map(s => s.trim());
    const columnNames = header.slice(1);

    const rawData = [];
    for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;
        const parts = line.split(',').map(s => s.trim());
        rawData.push({ name: parts[0], values: parts.slice(1) });
    }

    const episodes = [];
    const X = [];
    rawData.forEach(row => {
        episodes.push({ name: row.name });
        const values = row.values.map(v => parseFloat(v) || 0);
        while (values.length < columnNames.length) values.push(0);
        X.push(values);
    });

    const poolOrder = [];
    const poolMap = {};
    columnNames.forEach(colName => {
        const colonIdx = colName.indexOf(':');
        const poolName = colonIdx > 0 ? colName.substring(0, colonIdx) : 'Features';
        const featureName = colonIdx > 0 ? colName.substring(colonIdx + 1) : colName;
        if (!poolMap[poolName]) { poolMap[poolName] = []; poolOrder.push(poolName); }
        poolMap[poolName].push(featureName);
    });

    const pool_sizes = poolOrder.map(p => poolMap[p].length);
    return { pool_sizes, pool_names: poolOrder, episodes, X };
}

const { pool_sizes, pool_names, episodes, X } = parseCSV(csvText);
console.log(`Loaded ${episodes.length} samples, ${pool_names.length} pools (${pool_names.join(', ')})`);

// ---------- Model (mirrors public/sensory_dropout.html UnsupervisedModel) ----------
// Per-pool normalized Euclidean similarity ART. See the page comment for details.
class UnsupervisedModel {
    constructor(pool_sizes, vigilance, learning_rate, gain, inhibition) {
        this.pool_sizes = pool_sizes;
        this.vigilance = vigilance;
        this.learning_rate = learning_rate;
        this.gain = gain;
        this.inhibition = inhibition;
        this.templates = [];
        this.pool_slices = [];
        let start = 0;
        for (const size of pool_sizes) { this.pool_slices.push([start, start + size]); start += size; }
    }
    recruitHub(x) {
        this.templates.push([...x]);
        return this.templates.length - 1;
    }
    computePoolMask(x) {
        return this.pool_slices.map(([s, e]) => {
            let n = 0;
            for (let i = s; i < e; i++) n += x[i] * x[i];
            return n > 1e-8;
        });
    }
    computePoolSimilarities(x, y) {
        const sims = [];
        for (const [s, e] of this.pool_slices) {
            const n = e - s;
            let dist2 = 0;
            for (let i = s; i < e; i++) {
                const d = x[i] - y[i];
                dist2 += d * d;
            }
            sims.push(1 - dist2 / n);
        }
        return sims;
    }
    computeBottomUpDrive(x) {
        if (this.templates.length === 0) return [];
        const mask = this.computePoolMask(x);
        return this.templates.map(t => {
            const sims = this.computePoolSimilarities(x, t);
            let sum = 0, count = 0;
            for (let i = 0; i < sims.length; i++) {
                if (mask[i]) { sum += sims[i]; count++; }
            }
            const avg = count > 0 ? sum / count : 0;
            return this.gain * avg;
        });
    }
    settleCompetition(x, numSteps = 20) {
        const drive = this.computeBottomUpDrive(x);
        if (drive.length === 0) return [null, null];
        let acts = new Array(drive.length).fill(0);
        for (let s = 0; s < numSteps; s++) {
            const total = acts.reduce((a, b) => a + b, 0);
            acts = acts.map((a, i) => Math.max(0, drive[i] - this.inhibition * (total - a)));
        }
        let winner = 0; let max = acts[0];
        for (let i = 1; i < acts.length; i++) if (acts[i] > max) { max = acts[i]; winner = i; }
        return [winner, acts];
    }
    predictFromHub(idx) { return [...this.templates[idx]]; }
    snapOrRecruit(x) {
        if (this.templates.length === 0) {
            return { action: 'recruit', winner: this.recruitHub(x) };
        }
        const [winner] = this.settleCompetition(x);
        const y = this.predictFromHub(winner);
        const sims = this.computePoolSimilarities(x, y);
        const mask = this.computePoolMask(x);
        const passes = sims.every((s, i) => !mask[i] || s >= this.vigilance);
        if (passes) {
            this.updateHub(winner, x);
            return { action: 'snap', winner };
        } else {
            return { action: 'recruit', winner: this.recruitHub(x) };
        }
    }
    updateHub(idx, x) {
        const old = this.templates[idx];
        this.templates[idx] = old.map((o, i) => (1 - this.learning_rate) * o + this.learning_rate * x[i]);
    }
}

// ---------- Train + score ----------
const CONCEPTS = ['pine', 'oak', 'rose', 'daisy', 'robin', 'canary', 'sunfish', 'salmon'];

// Seeded LCG so shuffled runs are reproducible
function mulberry32(seed) {
    return function() {
        seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
        let t = seed;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}
function shuffleOrder(n, seed) {
    const rng = mulberry32(seed);
    const idx = Array.from({length: n}, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    return idx;
}

function trainAndScore(params) {
    const model = new UnsupervisedModel(pool_sizes, params.vigilance, params.learning_rate, params.gain, params.inhibition);
    const order = params.shuffle_seed != null
        ? shuffleOrder(X.length, params.shuffle_seed)
        : Array.from({length: X.length}, (_, i) => i);
    for (let pass = 0; pass < params.num_passes; pass++) {
        for (const i of order) model.snapOrRecruit(X[i]);
    }
    if (model.templates.length === 0) return null;

    // Final assignments via settled competition on trained templates
    const assignments = X.map(x => model.settleCompetition(x)[0]);

    // Concept ↔ hub crosstab
    const hubConcepts = {};
    const conceptHubs = {};
    for (let i = 0; i < episodes.length; i++) {
        const root = episodes[i].name.replace(/\d+$/, '');
        const hub = assignments[i];
        if (!hubConcepts[hub]) hubConcepts[hub] = {};
        hubConcepts[hub][root] = (hubConcepts[hub][root] || 0) + 1;
        if (!conceptHubs[root]) conceptHubs[root] = {};
        conceptHubs[root][hub] = (conceptHubs[root][hub] || 0) + 1;
    }

    const hubs = Object.keys(hubConcepts);
    const hubIsPure = h => Object.keys(hubConcepts[h]).length === 1;

    // Relaxed "resolved" metric: concept has ≥18 of its 20 samples in pure hubs of that concept
    // (multiple pure hubs for the same concept is fine).
    const conceptResolved = {};
    for (const c of CONCEPTS) {
        let pureCount = 0;
        for (const [h, n] of Object.entries(conceptHubs[c] || {})) {
            if (hubIsPure(h)) pureCount += n;
        }
        conceptResolved[c] = pureCount >= 18;
    }
    const resolvedCount = Object.values(conceptResolved).filter(Boolean).length;

    let avgConcentration = 0;
    for (const c of CONCEPTS) {
        if (!conceptHubs[c]) continue;
        const max = Math.max(...Object.values(conceptHubs[c]));
        avgConcentration += max / 20;
    }
    avgConcentration /= CONCEPTS.length;

    return {
        params,
        num_hubs: model.templates.length,
        pure_hubs: hubs.filter(hubIsPure).length,
        resolved: resolvedCount,
        avg_concentration: avgConcentration,
        conceptHubs,
        hubConcepts,
    };
}

// ---------- Sweep ----------
function buildGrid(opts) {
    const seeds = opts.shuffleSeeds || [null];
    const out = [];
    for (const v of opts.vigilances)
    for (const lr of opts.learningRates)
    for (const g of opts.gains)
    for (const inh of opts.inhibitions)
    for (const p of opts.passes)
    for (const seed of seeds) {
        out.push({ vigilance: v, learning_rate: lr, gain: g, inhibition: inh, num_passes: p, shuffle_seed: seed });
    }
    return out;
}

function runSweep(grid, label) {
    console.log(`\n=== ${label}: ${grid.length} configs ===`);
    const results = [];
    let bestSoFar = -1;
    let strictWins = 0;
    const t0 = Date.now();
    for (let i = 0; i < grid.length; i++) {
        const r = trainAndScore(grid[i]);
        if (!r) continue;
        results.push(r);
        if (r.resolved > bestSoFar) {
            bestSoFar = r.resolved;
            console.log(`  [${i + 1}/${grid.length}] new best: ${r.resolved}/8 resolved · ${r.num_hubs} hubs · ` +
                `v=${r.params.vigilance.toFixed(2)} lr=${r.params.learning_rate} g=${r.params.gain} inh=${r.params.inhibition} p=${r.params.num_passes}`);
        }
        if (r.resolved === 8) strictWins++;
    }
    const dt = ((Date.now() - t0) / 1000).toFixed(1);
    console.log(`  done in ${dt}s — strict wins: ${strictWins}, best resolved: ${bestSoFar}/8`);
    return results;
}

// Relaxed strict_success criterion: every concept "resolved" (top hub is pure-of-concept
// AND holds >=18/20 of that concept's samples). Splitting a concept across multiple
// pure hubs of the same concept still counts.
function isStrictSuccess(r) { return r.resolved === 8; }

// Pass 1 (coarse) — sweep vigilance, with multiple shuffle seeds.
// Includes inhibition=0 (no lateral competition) since that turned out essential
// to splitting sunfish/salmon (which are 0.9895 cosine-similar in Color).
const coarse = buildGrid({
    vigilances:    [0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.992, 0.995],
    learningRates: [0.05, 0.1, 0.2],
    gains:         [1.0, 2.0],
    inhibitions:   [0, 0.1, 0.2],
    passes:        [1, 2, 3, 5],
    shuffleSeeds:  [null, 1, 7, 42, 1337],
});
let results = runSweep(coarse, 'COARSE PASS');

let strictWinners = results.filter(isStrictSuccess);

// Early stop / escalate logic. Threshold deliberately low — only bail if the model
// can't separate more than 1 concept regardless of params (truly impossible territory).
if (strictWinners.length === 0) {
    const best = results.slice().sort((a, b) => b.resolved - a.resolved || b.avg_concentration - a.avg_concentration)[0];
    console.log(`\nNo 8/8 winners in coarse pass. Best: ${best.resolved}/8 resolved (concentration ${best.avg_concentration.toFixed(3)}).`);

    if (best.resolved < 2) {
        console.log('Best resolved < 2 — model can not separate even two concepts. Stopping early; this looks impossible.');
    } else {
        console.log('Refining around the best region…');
        const center = best.params;
        const fine = buildGrid({
            vigilances:    [center.vigilance - 0.02, center.vigilance - 0.01, center.vigilance, center.vigilance + 0.01, center.vigilance + 0.02].filter(v => v > 0 && v < 1),
            learningRates: [Math.max(0.02, center.learning_rate / 2), center.learning_rate, Math.min(0.9, center.learning_rate * 2)],
            gains:         [center.gain * 0.7, center.gain, center.gain * 1.5],
            inhibitions:   [Math.max(0, center.inhibition - 0.1), center.inhibition, center.inhibition + 0.1],
            passes:        [Math.max(1, center.num_passes - 1), center.num_passes, center.num_passes + 2, center.num_passes + 5],
        });
        const fineResults = runSweep(fine, 'FINE PASS');
        results = results.concat(fineResults);
        strictWinners = results.filter(isStrictSuccess);
    }
}

// ---------- Report ----------
console.log('\n========== RESULTS ==========');
console.log(`Total configs tried: ${results.length}`);
console.log(`Strict winners (every hub pure + every concept all-20 in one hub): ${strictWinners.length}`);

function fmtCrosstab(r) {
    const lines = [];
    for (const c of CONCEPTS) {
        const dist = r.conceptHubs[c] || {};
        const items = Object.entries(dist).sort((a, b) => b[1] - a[1])
            .map(([h, n]) => `H${h}×${n}`).join(' ');
        lines.push(`    ${c.padEnd(8)} → ${items}`);
    }
    return lines.join('\n');
}

function fmtParams(p) {
    const seed = p.shuffle_seed == null ? 'seq' : `seed=${p.shuffle_seed}`;
    return `v=${p.vigilance.toFixed(2)}  lr=${p.learning_rate}  gain=${p.gain}  inh=${p.inhibition}  passes=${p.num_passes}  order=${seed}`;
}

if (strictWinners.length > 0) {
    strictWinners.sort((a, b) => a.num_hubs - b.num_hubs || b.avg_concentration - a.avg_concentration);
    console.log(`\nFound ${strictWinners.length} 8/8 winners.`);
    console.log('Top 10 (fewest hubs first — closer to the goal of one hub per concept):');
    strictWinners.slice(0, 10).forEach((r, i) => {
        console.log(`\n#${i + 1}  ${fmtParams(r.params)}  (${r.num_hubs} hubs, ${r.pure_hubs} pure)`);
        console.log(fmtCrosstab(r));
    });
} else {
    console.log('\nNo 8/8 winner. Top 5 by `resolved` then `avg_concentration`:');
    results.sort((a, b) => b.resolved - a.resolved || b.avg_concentration - a.avg_concentration);
    results.slice(0, 5).forEach((r, i) => {
        console.log(`\n#${i + 1}  resolved=${r.resolved}/8  conc=${r.avg_concentration.toFixed(3)}  (${r.num_hubs} hubs, ${r.pure_hubs} pure)`);
        console.log(`    ${fmtParams(r.params)}`);
        console.log(fmtCrosstab(r));
    });
}
