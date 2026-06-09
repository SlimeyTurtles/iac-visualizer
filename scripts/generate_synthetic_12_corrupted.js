// Generates a 14-prototype dataset with deliberately corrupted ("super wrong")
// samples mixed in. Used by public/synthetic_12_decay.html to demonstrate that
// templates of hubs recruited by garbage samples decay away when an EWMA-decay
// term is added to the ART update rule.
//
// Output: 14 clean prototypes × CLEAN_PER_PROTO samples (default 20)
//       + 14 prototypes × BAD_PER_PROTO samples with uniformly random feature
//         values (default 2). Corrupted samples keep their concept-name prefix
//         so you can still see *which* concept they were supposed to be, with
//         a `_BAD` suffix so they're distinguishable in the visualization.
//
// Re-generate by running:  node scripts/generate_synthetic_12_corrupted.js

const fs = require('fs');
const path = require('path');
const XLSX = require('xlsx');

const POOLS = [
    { name: 'Color',      features: ['red', 'green', 'blue'],                       simplex: false },
    { name: 'Locomotion', features: ['ground', 'water', 'air'],                     simplex: true  },
    { name: 'Covering',   features: ['smoothness', 'hardness', 'dryness'],          simplex: false },
    { name: 'BodyParts',  features: ['legs', 'wings', 'fins', 'leaves', 'petals'],  simplex: false },
    { name: 'Habitat',    features: ['forest', 'water', 'garden', 'urban'],         simplex: true  },
];

const PROTOTYPES = {
    pine:    { Color: [0.15, 0.55, 0.15], Locomotion: [1.00, 0.00, 0.00], Covering: [0.15, 0.90, 0.85], BodyParts: [0.00, 0.00, 0.00, 0.70, 0.00], Habitat: [0.85, 0.00, 0.10, 0.05] },
    oak:     { Color: [0.55, 0.30, 0.10], Locomotion: [1.00, 0.00, 0.00], Covering: [0.15, 0.90, 0.85], BodyParts: [0.00, 0.00, 0.00, 1.00, 0.00], Habitat: [0.70, 0.00, 0.20, 0.10] },
    tree:    { Color: [0.10, 0.15, 0.08], Locomotion: [1.00, 0.00, 0.00], Covering: [0.15, 0.90, 0.85], BodyParts: [0.00, 0.00, 0.00, 0.85, 0.00], Habitat: [0.78, 0.00, 0.15, 0.07] },

    rose:    { Color: [0.90, 0.15, 0.20], Locomotion: [1.00, 0.00, 0.00], Covering: [0.55, 0.40, 0.65], BodyParts: [0.00, 0.00, 0.00, 0.50, 0.85], Habitat: [0.05, 0.00, 0.80, 0.15] },
    daisy:   { Color: [0.90, 0.90, 0.40], Locomotion: [1.00, 0.00, 0.00], Covering: [0.55, 0.40, 0.65], BodyParts: [0.00, 0.00, 0.00, 0.50, 0.85], Habitat: [0.00, 0.00, 0.85, 0.15] },
    flower:  { Color: [0.15, 0.15, 0.10], Locomotion: [1.00, 0.00, 0.00], Covering: [0.55, 0.40, 0.65], BodyParts: [0.00, 0.00, 0.00, 0.50, 0.85], Habitat: [0.03, 0.00, 0.83, 0.15] },

    plant:   { Color: [0.06, 0.08, 0.06], Locomotion: [1.00, 0.00, 0.00], Covering: [0.35, 0.65, 0.75], BodyParts: [0.00, 0.00, 0.00, 0.40, 0.30], Habitat: [0.35, 0.00, 0.40, 0.25] },

    robin:   { Color: [0.75, 0.30, 0.20], Locomotion: [0.30, 0.00, 0.70], Covering: [0.60, 0.25, 0.95], BodyParts: [0.50, 0.80, 0.00, 0.00, 0.00], Habitat: [0.25, 0.00, 0.40, 0.35] },
    canary:  { Color: [0.95, 0.85, 0.10], Locomotion: [0.20, 0.00, 0.80], Covering: [0.60, 0.30, 0.95], BodyParts: [0.40, 0.80, 0.00, 0.00, 0.00], Habitat: [0.20, 0.00, 0.20, 0.60] },
    bird:    { Color: [0.12, 0.10, 0.08], Locomotion: [0.25, 0.00, 0.75], Covering: [0.60, 0.28, 0.95], BodyParts: [0.45, 0.80, 0.00, 0.00, 0.00], Habitat: [0.23, 0.00, 0.30, 0.48] },

    sunfish: { Color: [0.40, 0.60, 0.75], Locomotion: [0.00, 1.00, 0.00], Covering: [0.85, 0.20, 0.05], BodyParts: [0.00, 0.00, 0.70, 0.00, 0.00], Habitat: [0.00, 1.00, 0.00, 0.00] },
    salmon:  { Color: [0.85, 0.85, 0.90], Locomotion: [0.00, 1.00, 0.00], Covering: [0.85, 0.20, 0.05], BodyParts: [0.00, 0.00, 0.80, 0.00, 0.00], Habitat: [0.00, 1.00, 0.00, 0.00] },
    fish:    { Color: [0.10, 0.15, 0.20], Locomotion: [0.00, 1.00, 0.00], Covering: [0.85, 0.20, 0.05], BodyParts: [0.00, 0.00, 0.75, 0.00, 0.00], Habitat: [0.00, 1.00, 0.00, 0.00] },

    animal:  { Color: [0.08, 0.08, 0.08], Locomotion: [0.10, 0.50, 0.40], Covering: [0.65, 0.20, 0.45], BodyParts: [0.20, 0.35, 0.30, 0.00, 0.00], Habitat: [0.12, 0.50, 0.18, 0.20] },
};

const CONFIG = {
    clean_per_proto: 20,
    bad_per_proto: 2,       // each concept gets this many "super wrong" samples mixed in
    noise_std: 0.025,
    seed: 42,
};

function mulberry32(seed) {
    return function () {
        seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
        let t = seed;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}
function normal(rng, mean, std) {
    const u1 = Math.max(rng(), 1e-12);
    const u2 = rng();
    return mean + std * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}
const clip = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

function cleanSample(rng, proto) {
    const sample = {};
    for (const pool of POOLS) {
        const noisy = proto[pool.name].map(v => clip(v + normal(rng, 0, CONFIG.noise_std), 0, 1));
        if (pool.simplex) {
            const sum = noisy.reduce((s, v) => s + v, 0);
            sample[pool.name] = sum > 1e-8 ? noisy.map(v => v / sum) : noisy;
        } else {
            sample[pool.name] = noisy;
        }
    }
    return sample;
}

// "Super wrong": every feature drawn uniformly random in [0, 1]. Simplex pools
// renormalized to sum to 1. The resulting vector has no relationship to its
// label's prototype.
function badSample(rng) {
    const sample = {};
    for (const pool of POOLS) {
        const raw = pool.features.map(() => rng());
        if (pool.simplex) {
            const sum = raw.reduce((s, v) => s + v, 0);
            sample[pool.name] = sum > 1e-8 ? raw.map(v => v / sum) : raw;
        } else {
            sample[pool.name] = raw;
        }
    }
    return sample;
}

function generate() {
    const rng = mulberry32(CONFIG.seed);
    const rows = [];
    for (const [protoName, proto] of Object.entries(PROTOTYPES)) {
        for (let i = 0; i < CONFIG.clean_per_proto; i++) {
            rows.push({ name: `${protoName}${i + 1}`, sample: cleanSample(rng, proto), corrupted: false });
        }
        for (let j = 0; j < CONFIG.bad_per_proto; j++) {
            rows.push({ name: `${protoName}_BAD${j + 1}`, sample: badSample(rng), corrupted: true });
        }
    }
    return rows;
}

const rows = generate();
const header = ['Name', ...POOLS.flatMap(p => p.features.map(f => `${p.name}:${f}`))];
const csvLines = [header.join(',')];
for (const row of rows) {
    const vals = [row.name];
    for (const pool of POOLS) for (let i = 0; i < pool.features.length; i++) vals.push(row.sample[pool.name][i].toFixed(4));
    csvLines.push(vals.join(','));
}
const csv = csvLines.join('\n');

const out = `// Synthetic 14-prototype dataset with deliberately corrupted samples ("_BAD" suffix).
// Generated by scripts/generate_synthetic_12_corrupted.js.
// ${rows.length} rows: 14 concepts × ${CONFIG.clean_per_proto} clean + ${CONFIG.bad_per_proto} corrupted.
// Corrupted samples have uniformly random feature values (still respecting
// simplex constraints for Locomotion/Habitat). They keep their concept-name
// prefix so it's visible *which* concept the corruption was labeled as.
// Noise std on clean samples = ${CONFIG.noise_std}.
// Re-generate by running:  node scripts/generate_synthetic_12_corrupted.js

window.SYNTHETIC_12_CORRUPTED_CSV = \`
${csv}\`;
`;

const jsPath = path.join(__dirname, '..', 'public', 'synthetic_12_corrupted_data.js');
fs.writeFileSync(jsPath, out);

const wb = XLSX.utils.book_new();
const sampleRows = [header];
for (const row of rows) {
    const vals = [row.name];
    for (const pool of POOLS) for (let i = 0; i < pool.features.length; i++) vals.push(row.sample[pool.name][i]);
    sampleRows.push(vals);
}
XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet(sampleRows), 'Samples');

const protoNames = Object.keys(PROTOTYPES);
const protoSheet = [['Pool', 'Type', 'Feature', ...protoNames]];
for (const pool of POOLS) {
    pool.features.forEach((feat, idx) => {
        protoSheet.push([
            pool.name,
            pool.simplex ? 'simplex (sum=1)' : 'independent (0–1)',
            feat,
            ...protoNames.map(p => PROTOTYPES[p][pool.name][idx]),
        ]);
    });
}
protoSheet.push([], ['--- Generation config ---']);
protoSheet.push(['clean_per_proto', CONFIG.clean_per_proto]);
protoSheet.push(['bad_per_proto (random-feature samples mixed in)', CONFIG.bad_per_proto]);
protoSheet.push(['noise_std (per-feature normal noise, clean only)', CONFIG.noise_std]);
protoSheet.push(['seed (mulberry32)', CONFIG.seed]);
XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet(protoSheet), 'Prototypes');

const xlsxPath = path.join(__dirname, '..', 'public', 'synthetic_12_corrupted_data.xlsx');
XLSX.writeFile(wb, xlsxPath);

const badCount = rows.filter(r => r.corrupted).length;
console.log(`Wrote ${rows.length} rows (${rows.length - badCount} clean + ${badCount} corrupted) to:`);
console.log(`  - ${jsPath}`);
console.log(`  - ${xlsxPath}`);
