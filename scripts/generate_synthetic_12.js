// Generates a synthetic dataset of 12 prototypes (8 species + 4 generic superordinates)
// for the Hierarchy Emergence ART demo. Writes public/synthetic_12_data.js.
//
// Each pool is declared as either:
//   - simplex: features sum to 1 (e.g. Locomotion, Habitat — a thing is air OR water OR ground,
//     not several at once except as a probability mass)
//   - non-simplex: each feature independently in [0,1] (e.g. Color — RGB components are
//     not constrained to sum to anything; an object can be unsaturated 0,0,0 or super-bright 1,1,1)
//
// Noise model: normal(0, NOISE_STD) per feature, then per-pool handling:
//   - simplex: clip to >= 0, renormalize to sum=1
//   - non-simplex: clip to [0, 1]
//
// Dropout: each pool has DROPOUT_PROB chance of being entirely zeroed out (simulating
// a missing-modality sample), mirroring the original sensory_data.js dataset.

const fs = require('fs');
const path = require('path');
const XLSX = require('xlsx');

// ---- Pool schema ----
const POOLS = [
    { name: 'Color',      features: ['red', 'green', 'blue'],                       simplex: false },
    { name: 'Locomotion', features: ['ground', 'water', 'air'],                     simplex: true  },
    { name: 'Covering',   features: ['smoothness', 'hardness', 'dryness'],          simplex: false },
    { name: 'BodyParts',  features: ['legs', 'wings', 'fins', 'leaves', 'petals'],  simplex: false },
    { name: 'Habitat',    features: ['forest', 'water', 'garden', 'urban'],         simplex: true  },
];

// ---- 12 prototypes (8 species + 4 generic superordinates) ----
// Generic prototypes carry the same shape-defining features as their species but with
// the *distinguishing* features (mostly Color) damped toward zero — a "dull" version
// that says "I'm a flower" but doesn't commit to red or yellow.
// Species prototypes are clearly distinct in their discriminating pool (Color, mostly).
// Generic superordinate prototypes have COLOR damped toward zero — the "dulled" version
// that says "I'm a flower" but doesn't commit to red or yellow. They keep their kingdom's
// non-color features (petals for flower, fins for fish, etc.).
const PROTOTYPES = {
    pine:    { Color: [0.15, 0.55, 0.15], Locomotion: [1.00, 0.00, 0.00], Covering: [0.15, 0.90, 0.85], BodyParts: [0.00, 0.00, 0.00, 0.70, 0.00], Habitat: [0.85, 0.00, 0.10, 0.05] },
    oak:     { Color: [0.55, 0.30, 0.10], Locomotion: [1.00, 0.00, 0.00], Covering: [0.15, 0.90, 0.85], BodyParts: [0.00, 0.00, 0.00, 1.00, 0.00], Habitat: [0.70, 0.00, 0.20, 0.10] },
    tree:    { Color: [0.10, 0.15, 0.08], Locomotion: [1.00, 0.00, 0.00], Covering: [0.18, 0.85, 0.85], BodyParts: [0.00, 0.00, 0.00, 0.35, 0.00], Habitat: [0.65, 0.00, 0.20, 0.15] },

    rose:    { Color: [0.90, 0.15, 0.20], Locomotion: [1.00, 0.00, 0.00], Covering: [0.55, 0.40, 0.65], BodyParts: [0.00, 0.00, 0.00, 0.50, 0.85], Habitat: [0.05, 0.00, 0.80, 0.15] },
    daisy:   { Color: [0.90, 0.90, 0.40], Locomotion: [1.00, 0.00, 0.00], Covering: [0.55, 0.40, 0.65], BodyParts: [0.00, 0.00, 0.00, 0.50, 0.85], Habitat: [0.00, 0.00, 0.85, 0.15] },
    flower:  { Color: [0.15, 0.15, 0.10], Locomotion: [1.00, 0.00, 0.00], Covering: [0.50, 0.35, 0.55], BodyParts: [0.00, 0.00, 0.00, 0.35, 0.40], Habitat: [0.10, 0.00, 0.70, 0.20] },

    robin:   { Color: [0.75, 0.30, 0.20], Locomotion: [0.30, 0.00, 0.70], Covering: [0.60, 0.25, 0.95], BodyParts: [0.50, 0.80, 0.00, 0.00, 0.00], Habitat: [0.25, 0.00, 0.40, 0.35] },
    canary:  { Color: [0.95, 0.85, 0.10], Locomotion: [0.20, 0.00, 0.80], Covering: [0.60, 0.30, 0.95], BodyParts: [0.40, 0.80, 0.00, 0.00, 0.00], Habitat: [0.20, 0.00, 0.20, 0.60] },
    bird:    { Color: [0.12, 0.10, 0.08], Locomotion: [0.25, 0.00, 0.75], Covering: [0.55, 0.25, 0.90], BodyParts: [0.40, 0.65, 0.00, 0.00, 0.00], Habitat: [0.25, 0.00, 0.30, 0.45] },

    sunfish: { Color: [0.40, 0.60, 0.75], Locomotion: [0.00, 1.00, 0.00], Covering: [0.85, 0.20, 0.05], BodyParts: [0.00, 0.00, 0.70, 0.00, 0.00], Habitat: [0.00, 1.00, 0.00, 0.00] },
    salmon:  { Color: [0.85, 0.85, 0.90], Locomotion: [0.00, 1.00, 0.00], Covering: [0.85, 0.20, 0.05], BodyParts: [0.00, 0.00, 0.80, 0.00, 0.00], Habitat: [0.00, 1.00, 0.00, 0.00] },
    fish:    { Color: [0.10, 0.15, 0.20], Locomotion: [0.00, 1.00, 0.00], Covering: [0.80, 0.20, 0.10], BodyParts: [0.00, 0.00, 0.45, 0.00, 0.00], Habitat: [0.00, 1.00, 0.00, 0.00] },
};

const CONFIG = {
    samples_per_proto: 20,
    noise_std: 0.025,
    dropout_prob: 0.03,
    seed: 42,
};

// ---- Seeded RNG (mulberry32) ----
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
    // Box-Muller
    const u1 = Math.max(rng(), 1e-12);
    const u2 = rng();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return mean + std * z;
}
const clip = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// ---- The generator function ----
function generateSyntheticData({ pools, prototypes, samples_per_proto, noise_std, dropout_prob, seed }) {
    const rng = mulberry32(seed);
    const rows = [];

    for (const [protoName, proto] of Object.entries(prototypes)) {
        for (let i = 0; i < samples_per_proto; i++) {
            const sample = {};

            for (const pool of pools) {
                const noisy = proto[pool.name].map(v => clip(v + normal(rng, 0, noise_std), 0, 1));

                if (pool.simplex) {
                    const sum = noisy.reduce((s, v) => s + v, 0);
                    sample[pool.name] = sum > 1e-8 ? noisy.map(v => v / sum) : noisy;
                } else {
                    sample[pool.name] = noisy;
                }
            }

            // Dropout: zero out any pool with prob dropout_prob
            for (const pool of pools) {
                if (rng() < dropout_prob) {
                    sample[pool.name] = sample[pool.name].map(() => 0);
                }
            }

            rows.push({ name: `${protoName}${i + 1}`, sample });
        }
    }

    // Build CSV
    const header = ['Name'];
    for (const pool of pools) {
        for (const f of pool.features) header.push(`${pool.name}:${f}`);
    }
    const csvLines = [header.join(',')];
    for (const row of rows) {
        const vals = [row.name];
        for (const pool of pools) {
            for (let i = 0; i < pool.features.length; i++) {
                vals.push(row.sample[pool.name][i].toFixed(4));
            }
        }
        csvLines.push(vals.join(','));
    }
    return { csv: csvLines.join('\n'), rows };
}

// ---- Variant writer ----
// Writes both .js (for the page) and .xlsx (for download/inspection) for a single variant.
function writeVariant({ basename, csvVar, dropoutProb, label }) {
    const { csv, rows } = generateSyntheticData({
        pools: POOLS,
        prototypes: PROTOTYPES,
        samples_per_proto: CONFIG.samples_per_proto,
        noise_std: CONFIG.noise_std,
        dropout_prob: dropoutProb,
        seed: CONFIG.seed,
    });

    const out = `// Synthetic 12-prototype dataset — ${label}. Generated by scripts/generate_synthetic_12.js.
// 12 categories × ${CONFIG.samples_per_proto} samples = ${rows.length} rows.
// Categories: 8 species (pine, oak, rose, daisy, robin, canary, sunfish, salmon)
//           + 4 generic superordinates (tree, flower, bird, fish) with damped distinguishing features.
// Simplex pools (Locomotion, Habitat) sum to 1; others are independent 0-1.
// Noise std = ${CONFIG.noise_std}; dropout prob = ${dropoutProb}.
// Re-generate by running:  node scripts/generate_synthetic_12.js

window.${csvVar} = \`
${csv}\`;
`;
    const jsPath = path.join(__dirname, '..', 'public', `${basename}.js`);
    fs.writeFileSync(jsPath, out);

    const wb = XLSX.utils.book_new();

    const header = ['Name', ...POOLS.flatMap(p => p.features.map(f => `${p.name}:${f}`))];
    const sampleRows = [header];
    for (const row of rows) {
        const vals = [row.name];
        for (const pool of POOLS) {
            for (let i = 0; i < pool.features.length; i++) {
                vals.push(row.sample[pool.name][i]);
            }
        }
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
    protoSheet.push([]);
    protoSheet.push(['--- Generation config ---']);
    protoSheet.push(['variant', label]);
    protoSheet.push(['samples_per_proto', CONFIG.samples_per_proto]);
    protoSheet.push(['noise_std (per-feature normal noise)', CONFIG.noise_std]);
    protoSheet.push(['dropout_prob (per-pool zero-out chance)', dropoutProb]);
    protoSheet.push(['seed (mulberry32)', CONFIG.seed]);
    protoSheet.push([]);
    protoSheet.push(['--- Notes ---']);
    protoSheet.push(['Pool order in samples matches the order in the Pool column above.']);
    protoSheet.push(['Generic prototypes (tree, flower, bird, fish) have damped distinguishing-feature values.']);
    protoSheet.push(['Simplex pools are clipped to ≥0 then renormalized to sum=1 after noise. Non-simplex are clipped to [0,1].']);
    XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet(protoSheet), 'Prototypes');

    const xlsxPath = path.join(__dirname, '..', 'public', `${basename}.xlsx`);
    XLSX.writeFile(wb, xlsxPath);

    console.log(`[${label}] ${rows.length} rows:`);
    console.log(`  - ${jsPath}`);
    console.log(`  - ${xlsxPath}`);
}

// ---- Run both variants ----
writeVariant({
    basename: 'synthetic_12_data',
    csvVar: 'SYNTHETIC_12_CSV',
    dropoutProb: CONFIG.dropout_prob,
    label: 'with sensory dropout',
});
writeVariant({
    basename: 'synthetic_12_clean_data',
    csvVar: 'SYNTHETIC_12_CLEAN_CSV',
    dropoutProb: 0,
    label: 'no dropout (clean)',
});

console.log(`Categories: ${Object.keys(PROTOTYPES).join(', ')}`);
