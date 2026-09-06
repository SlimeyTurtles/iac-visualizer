// Headless checks for the ESCF page model (public/escf_model.js).
// Verifies the paper's headline behaviors across seeds:
//   Sim 1a: 14-category hierarchy (animal/plant, 4 basic kinds, 8 items)
//   Sim 2a: berries merge on sight, split by outcome, then read on sight
//   Sim 2b: stone/branch/bone assemble into one fends-off hub
//   Sim 3:  goal transform routes branch/stone (not daisy) to the weapon hub
// Usage: node scripts/tune_escf.js [numSeeds]
const ESCF = require('../public/escf_model.js');

const NUM_SEEDS = parseInt(process.argv[2] || '30', 10);

function dominantLabel(counts) {
    let best = null, bestN = 0, total = 0;
    for (const [k, v] of Object.entries(counts)) { total += v; if (v > bestN) { bestN = v; best = k; } }
    return { label: best, purity: bestN / total, total };
}

const BASIC = { robin: 'bird', canary: 'bird', salmon: 'fish', sunfish: 'fish', oak: 'tree', pine: 'tree', rose: 'flower', daisy: 'flower' };
const SUPER = { bird: 'animal', fish: 'animal', tree: 'plant', flower: 'plant' };

// Classify a phase-1 hub by member composition: item-level (one item), basic
// (one basic kind, >1 item), super (one superordinate, >1 basic kind), mixed.
function classifyHub(counts) {
    const items = Object.keys(counts).filter(k => BASIC[k]);
    if (items.length === 0) return null;
    if (items.length === 1) return { level: 'item', label: items[0] };
    const basics = new Set(items.map(i => BASIC[i]));
    if (basics.size === 1) return { level: 'basic', label: [...basics][0] };
    const supers = new Set([...basics].map(b => SUPER[b]));
    if (supers.size === 1) return { level: 'super', label: [...supers][0] };
    return { level: 'mixed', label: items.join('+') };
}

function runSeed(seed, verbose) {
    const { model, history } = ESCF.runAll({ seed });
    const res = { seed };

    // ---- Sim 1a: hierarchy from phase-1 members only ----
    const phase1Hubs = {};
    model.hubs.forEach(h => {
        const counts = {};
        h.members.filter(m => m.phase === 0).forEach(m => { counts[m.item] = (counts[m.item] || 0) + 1; });
        const n = Object.values(counts).reduce((a, b) => a + b, 0);
        if (n >= 2) phase1Hubs[h.id] = counts; // paper's >=2-trial reporting filter
    });
    const found = { super: new Set(), basic: new Set(), item: new Set(), mixed: [] };
    for (const counts of Object.values(phase1Hubs)) {
        const c = classifyHub(counts);
        if (!c) continue;
        if (c.level === 'mixed') found.mixed.push(c.label);
        else found[c.level].add(c.label);
    }
    res.supers = found.super.size;   // want 2
    res.basics = found.basic.size;   // want 4
    res.items = found.item.size;     // want 8
    res.mixed = found.mixed.length;
    res.hierarchyOK = res.supers === 2 && res.basics === 4 && res.items === 8 && res.mixed === 0;

    // ---- Sim 2a: merge on sight, split by outcome, read on sight ----
    // merge: pre-eating perceive trials of both berries land in one shared hub
    const berrySight = history.filter(h => h.phaseIdx === 1 && h.mode === 'perceive');
    const sightHubs = new Set(berrySight.map(h => h.winner));
    res.berriesMergedOnSight = sightHubs.size === 1;
    // split: post-eating there are two distinct outcome hubs
    let sweetHub = null, blandHub = null;
    model.hubs.forEach(h => {
        const o = model.outcomeOf(h);
        if (o && o.feature === 'sweet') { if (o.present) sweetHub = h.id; else blandHub = h.id; }
    });
    res.berriesSplit = sweetHub !== null && blandHub !== null && sweetHub !== blandHub;
    // read on sight: fresh perceptual probes route each berry to its outcome hub
    const pa = model.probe('berryA', model.p.actEngagement);
    const pb = model.probe('berryB', model.p.actEngagement);
    res.berryReadOnSight = res.berriesSplit && pa.winner === sweetHub && pb.winner === blandHub;

    // ---- Sim 2b: weapon assembly ----
    let weaponHub = null;
    model.hubs.forEach(h => {
        const o = model.outcomeOf(h);
        if (o && o.feature === 'fends-off' && o.present) weaponHub = h;
    });
    res.weaponAssembled = false;
    if (weaponHub) {
        const items = new Set(weaponHub.members.filter(m => m.phase === 2).map(m => m.item));
        res.weaponAssembled = ['stone', 'branch', 'bone'].every(i => items.has(i)) && !items.has('daisy');
    }

    // ---- Sim 3: goal probes ----
    const probes = history.filter(h => h.mode === 'probe');
    const byItem = {};
    probes.forEach(p => { byItem[`${p.item}:${p.goal ? 'on' : 'off'}`] = p; });
    const wid = weaponHub ? weaponHub.id : -1;
    const isPerceptHub = (rec) => rec && rec.winner !== wid;
    res.goalOK =
        isPerceptHub(byItem['branch:off']) && byItem['branch:on'] && byItem['branch:on'].winner === wid &&
        isPerceptHub(byItem['stone:off']) && byItem['stone:on'] && byItem['stone:on'].winner === wid &&
        isPerceptHub(byItem['daisy:off']) && byItem['daisy:on'] && byItem['daisy:on'].winner !== wid;

    if (verbose) {
        console.log(`\nseed ${seed}: super=${res.supers} basic=${res.basics} item=${res.items} mixed=${res.mixed}`);
        for (const [id, counts] of Object.entries(phase1Hubs)) {
            const c = classifyHub(counts);
            console.log(`  hub ${id} [${c ? c.level + ':' + c.label : '?'}]`, JSON.stringify(counts));
        }
        console.log(`  berries: merged=${res.berriesMergedOnSight} split=${res.berriesSplit} readOnSight=${res.berryReadOnSight}`);
        console.log(`  weapon assembled=${res.weaponAssembled}`);
        if (weaponHub) {
            const top = [];
            model.features.forEach((f, i) => {
                if (f.band === 'outcome') return;
                if (weaponHub.w[2 * i] > 0.1) top.push(`${f.name}:${weaponHub.w[2 * i].toFixed(2)}`);
            });
            console.log(`  weapon template (present units > 0.1): ${top.join(', ')}`);
        }
        console.log(`  goal: ${['branch', 'stone', 'daisy'].map(i => `${i} off->H${byItem[i + ':off'] && byItem[i + ':off'].winner} on->H${byItem[i + ':on'] && byItem[i + ':on'].winner}`).join(' | ')} (weapon=H${wid})`);
    }
    return res;
}

const tally = { hierarchyOK: 0, berriesMergedOnSight: 0, berriesSplit: 0, berryReadOnSight: 0, weaponAssembled: 0, goalOK: 0, allOK: 0 };
const supers = [], basics = [], items = [];
for (let s = 0; s < NUM_SEEDS; s++) {
    const r = runSeed(s, s === 0);
    for (const k of Object.keys(tally)) if (k !== 'allOK' && r[k]) tally[k]++;
    if (r.hierarchyOK && r.berriesMergedOnSight && r.berryReadOnSight && r.weaponAssembled && r.goalOK) tally.allOK++;
    supers.push(r.supers); basics.push(r.basics); items.push(r.items);
}
const avg = a => (a.reduce((x, y) => x + y, 0) / a.length).toFixed(2);
console.log(`\n=== ${NUM_SEEDS} seeds ===`);
console.log(`hierarchy (2/4/8, no mixed): ${tally.hierarchyOK}/${NUM_SEEDS}  (avg super=${avg(supers)} basic=${avg(basics)} item=${avg(items)})`);
console.log(`berries merged on sight:     ${tally.berriesMergedOnSight}/${NUM_SEEDS}`);
console.log(`berries split by outcome:    ${tally.berriesSplit}/${NUM_SEEDS}`);
console.log(`berries read on sight:       ${tally.berryReadOnSight}/${NUM_SEEDS}`);
console.log(`weapon hub assembled:        ${tally.weaponAssembled}/${NUM_SEEDS}`);
console.log(`goal probes correct:         ${tally.goalOK}/${NUM_SEEDS}`);
console.log(`ALL simulations pass:        ${tally.allOK}/${NUM_SEEDS}`);
