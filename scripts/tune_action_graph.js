// Headless checks for the Action Graph page model (public/action_graph_model.js).
// Trains on the modulo curriculum (A,B,C) -> ((A+B) mod C, 0, 0) and reports
// frozen-weights success before/after, broken down by optimal hop count, plus
// the learned step×action table.
// Usage: node scripts/tune_action_graph.js [episodes] [key=value ...]
//   e.g. node scripts/tune_action_graph.js 300 w0=0.012 decay=0.1 eta=0.05 budget=0.06 mode=step
const AG = require('../public/action_graph_model.js');

const EPISODES = parseInt(process.argv[2] || '300', 10);
const overrides = {};
for (const kv of process.argv.slice(3)) {
    const [k, v] = kv.split('=');
    overrides[k] = isNaN(+v) ? v : +v;
}
const SEED = overrides.seed ?? 1;
const RANGE = { aMin: 0, aMax: overrides.rA ?? 5, cMin: overrides.rCmin ?? 2, cMax: overrides.rC ?? 5 };
for (const k of ["rA", "rCmin", "rC", "seed"]) delete overrides[k];

function fmtByHops(byHops) {
    return Object.keys(byHops).sort((p, q) => (+p) - (+q))
        .map(h => `${h}:${byHops[h].ok}/${byHops[h].n}`).join('  ');
}

function run(opts, verbose) {
    const m = new AG.ActionGraphModel(opts);
    const tasks = m.allTasks(RANGE);
    const before = m.evaluate(tasks);
    const testBefore = m.evaluateTest(tasks);
    const rng = AG.mulberry32(SEED);
    const roll = [];
    for (let i = 0; i < EPISODES; i++) {
        const t = m.sampleTask(rng, RANGE);
        const r = m.episode(t.start, t.goal, { learn: true });
        roll.push(r.success ? 1 : 0);
    }
    const after = m.evaluate(tasks);
    const testAfter = m.evaluateTest(tasks);
    const late = roll.slice(-100).reduce((p, q) => p + q, 0) / Math.min(100, roll.length);
    console.log(`\nnodes=${m.n} edges=${m.nE} actions=${m.nA}  opts=${JSON.stringify(opts)}`);
    console.log(`  before: ${before.ok}/${before.n}  eff=${before.meanEfficiency.toFixed(2)}   ${fmtByHops(before.byHops)}`);
    console.log(`  after : ${after.ok}/${after.n}  eff=${after.meanEfficiency.toFixed(2)}   ${fmtByHops(after.byHops)}`);
    console.log(`  goal-free test: ${testBefore.ok}/${testBefore.n} -> ${testAfter.ok}/${testAfter.n}`);
    console.log(`  rolling success (last 100 train episodes): ${late.toFixed(2)}`);
    if (verbose) {
        const rows = m.mode === 'step' ? m.kMax : 1;
        for (let k = 0; k < rows; k++) {
            const row = [];
            for (let a = 0; a < m.nA; a++) row.push({ a, w: m.W[m.slot(k, a)] });
            row.sort((p, q) => q.w - p.w);
            const top = row.slice(0, 4).filter(d => d.w > 1e-4)
                .map(d => `${m.actions[d.a].short}=${d.w.toFixed(4)}`).join('  ');
            console.log(`  step ${k}: ${top || '(nothing learned)'}`);
        }
        // sample active-set size
        const t = AG.ActionGraphModel.modTask(4, 3, 3);
        const r = m.episode(t.start, t.goal, { learn: false });
        console.log(`  probe (4,3,3)->(${r.goal}): active=${r.active} plan=${r.plan.outcome} ` +
            `[${r.plan.actions.map(a => m.actions[a].short).join(' → ')}] bottleneck=${r.bridge.bottleneck.toFixed(3)} opt=${r.optHops}`);
    }
    return { before, after, late };
}

run(Object.assign({}, overrides), true);
