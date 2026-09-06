// ESCF — Emergent Semantic Cognition Framework (paper Sections 3-4).
// Data + model shared by public/escf.html (browser) and scripts/tune_escf.js (node).
//
// Duplex coding: every feature has a present unit and an absent unit. A feature
// that registers (activation >= theta) drives its present unit; every perceptual
// feature that does not register is coded absent (a positive fact, per the paper).
// Outcome features are undisclosed (both units 0) unless the item is acted on.
(function (root) {
    'use strict';

    // ---------- Feature inventory ----------
    // band: 'super' | 'basic' | 'detail' | 'outcome'
    // salience: per-feature; activation = min(1, salience + engagement + noise)
    const FEATURES = [
        // superordinate band — register at any engagement
        { name: 'self-moving', band: 'super' },
        { name: 'rooted', band: 'super' },
        // basic band — register at moderate engagement
        { name: 'wings', band: 'basic' },
        { name: 'feathers', band: 'basic' },
        { name: 'flies', band: 'basic' },
        { name: 'fins', band: 'basic' },
        { name: 'scales', band: 'basic' },
        { name: 'swims', band: 'basic' },
        { name: 'petals', band: 'basic' },
        { name: 'leaves', band: 'basic' },
        { name: 'woody', band: 'basic' },
        // detail band — register only at high engagement
        { name: 'red-breast', band: 'detail' },
        { name: 'hops', band: 'detail' },
        { name: 'worm-hunter', band: 'detail' },
        { name: 'yellow-body', band: 'detail' },
        { name: 'sings', band: 'detail' },
        { name: 'tame', band: 'detail' },
        { name: 'pink-flesh', band: 'detail' },
        { name: 'leaps', band: 'detail' },
        { name: 'upstream', band: 'detail' },
        { name: 'flat-body', band: 'detail' },
        { name: 'spiny-fin', band: 'detail' },
        { name: 'pond-dweller', band: 'detail' },
        { name: 'acorns', band: 'detail' },
        { name: 'broad-leaves', band: 'detail' },
        { name: 'thick-trunk', band: 'detail' },
        { name: 'needles', band: 'detail' },
        { name: 'cones', band: 'detail' },
        { name: 'evergreen', band: 'detail' },
        { name: 'thorns', band: 'detail' },
        { name: 'fragrant', band: 'detail' },
        { name: 'layered-petals', band: 'detail' },
        { name: 'white-rays', band: 'detail' },
        { name: 'thin-stem', band: 'detail' },
        { name: 'yellow-center', band: 'detail' },
        // berry / weapon perceptual features
        { name: 'round', band: 'basic' },
        { name: 'red-skin', band: 'basic' },
        { name: 'glossy', band: 'basic' },
        { name: 'juicy-look', band: 'basic' },
        { name: 'speckled', band: 'detail', salience: -0.06 }, // faint marker
        { name: 'hard', band: 'basic' },
        { name: 'rigid', band: 'basic' },
        { name: 'graspable', band: 'basic' },
        { name: 'hefty', band: 'basic' },
        { name: 'long', band: 'basic' },
        { name: 'grey', band: 'detail' },
        { name: 'bark-brown', band: 'detail' },
        { name: 'bone-white', band: 'detail' },
        { name: 'smooth', band: 'detail' },
        // outcome band — only observable by acting
        { name: 'sweet', band: 'outcome' },
        { name: 'fends-off', band: 'outcome' },
    ];

    const ITEMS = {
        // Simulation 1 corpus (Rumelhart & Todd / Rogers & McClelland items)
        robin:   ['self-moving', 'wings', 'feathers', 'flies', 'red-breast', 'hops', 'worm-hunter'],
        canary:  ['self-moving', 'wings', 'feathers', 'flies', 'yellow-body', 'sings', 'tame'],
        salmon:  ['self-moving', 'fins', 'scales', 'swims', 'pink-flesh', 'leaps', 'upstream'],
        sunfish: ['self-moving', 'fins', 'scales', 'swims', 'flat-body', 'spiny-fin', 'pond-dweller'],
        oak:     ['rooted', 'leaves', 'woody', 'acorns', 'broad-leaves', 'thick-trunk'],
        pine:    ['rooted', 'leaves', 'woody', 'needles', 'cones', 'evergreen'],
        rose:    ['rooted', 'petals', 'leaves', 'thorns', 'fragrant', 'layered-petals'],
        daisy:   ['rooted', 'petals', 'leaves', 'white-rays', 'thin-stem', 'yellow-center'],
        // Simulation 2a: look-alike berries; only 'speckled' (faint) tells them apart
        berryA:  ['rooted', 'round', 'red-skin', 'glossy', 'juicy-look', 'speckled'],
        berryB:  ['rooted', 'round', 'red-skin', 'glossy', 'juicy-look'],
        // Simulation 2b: perceptually unlike, all afford fending off
        stone:   ['hard', 'rigid', 'graspable', 'hefty', 'round', 'grey', 'smooth'],
        branch:  ['rooted', 'woody', 'leaves', 'hard', 'rigid', 'graspable', 'hefty', 'long', 'bark-brown'],
        bone:    ['hard', 'rigid', 'graspable', 'hefty', 'long', 'bone-white', 'smooth'],
    };

    // Outcomes disclosed by acting: [featureName, present?]
    const OUTCOMES = {
        berryA: ['sweet', true],
        berryB: ['sweet', false],
        stone:  ['fends-off', true],
        branch: ['fends-off', true],
        bone:   ['fends-off', true],
        daisy:  ['fends-off', false],
    };

    const DEFAULTS = {
        rho: 0.90,          // vigilance
        theta: 0.30,        // registration threshold
        sigma: 0.02,        // input noise
        absentAct: 0.06,    // activation of an absent unit
        outcomeAct: 0.85,   // activation of a disclosed outcome unit
        saliences: { super: 0.36, basic: 0.16, detail: 0.06 },
        engagementLevels: [0.0, 0.06, 0.18, 0.34, 0.46],
        actEngagement: 0.46, // engagement when acting on an item
        repsPerLevel: 6,
        goalSharpness: 2,   // exponent of the goal lens g = (w/max)^k
        seed: 42,
    };

    // ---------- utilities ----------
    function mulberry32(seed) {
        let a = seed >>> 0;
        return function () {
            a |= 0; a = (a + 0x6D2B79F5) | 0;
            let t = Math.imul(a ^ (a >>> 15), 1 | a);
            t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
    }
    function gauss(rand) {
        const u = Math.max(rand(), 1e-9), v = rand();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }
    function cosineOn(a, b, idxs) {
        let dot = 0, na = 0, nb = 0;
        for (const i of idxs) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
        if (na < 1e-12 || nb < 1e-12) return 0;
        return dot / Math.sqrt(na * nb);
    }

    // ---------- model ----------
    class ESCFModel {
        constructor(params) {
            this.p = Object.assign({}, DEFAULTS, params || {});
            this.p.saliences = Object.assign({}, DEFAULTS.saliences, (params || {}).saliences || {});
            this.rand = mulberry32(this.p.seed);
            this.features = FEATURES;
            this.nF = FEATURES.length;
            this.dim = 2 * this.nF; // [present_i, absent_i] pairs
            this.perceptualIdx = [];
            this.outcomeIdx = [];
            FEATURES.forEach((f, i) => {
                const pair = [2 * i, 2 * i + 1];
                if (f.band === 'outcome') this.outcomeIdx.push(...pair);
                else this.perceptualIdx.push(...pair);
            });
            this.hubs = []; // {w: Float64Array, n, members: [{item, phase}], id}
        }

        salienceOf(f) {
            return f.salience !== undefined ? f.salience : this.p.saliences[f.band];
        }

        // Build the duplex input for an item at an engagement level.
        // opts.disclose: outcome disclosed (acted-on trial)
        buildInput(item, engagement, opts) {
            opts = opts || {};
            const has = new Set(ITEMS[item]);
            const x = new Float64Array(this.dim);
            const cleared = [];
            const activations = new Float64Array(this.nF);
            this.features.forEach((f, i) => {
                if (f.band === 'outcome') {
                    if (opts.disclose && OUTCOMES[item] && OUTCOMES[item][0] === f.name) {
                        const present = OUTCOMES[item][1];
                        x[2 * i + (present ? 0 : 1)] = this.p.outcomeAct;
                        activations[i] = this.p.outcomeAct;
                        cleared.push({ feature: f.name, present, outcome: true });
                    }
                    return;
                }
                const a = Math.min(1, this.salienceOf(f) + engagement + gauss(this.rand) * this.p.sigma);
                if (has.has(f.name) && a >= this.p.theta) {
                    x[2 * i] = a;
                    activations[i] = a;
                    cleared.push({ feature: f.name, present: true });
                } else {
                    // not registered (or genuinely lacking) -> coded as observed-absent
                    x[2 * i + 1] = this.p.absentAct;
                    activations[i] = 0;
                }
            });
            return { x, cleared, activations };
        }

        matchAll(x) {
            return this.hubs.map(h => cosineOn(x, h.w, this.perceptualIdx));
        }

        outcomeOf(hub) {
            // returns {feature, present} if the hub carries outcome information
            for (let i = 0; i < this.nF; i++) {
                if (this.features[i].band !== 'outcome') continue;
                const p = hub.w[2 * i], a = hub.w[2 * i + 1];
                if (p > 0.05 || a > 0.05) return { feature: this.features[i].name, present: p >= a };
            }
            return null;
        }

        recruit(x, item, phase) {
            const hub = { id: this.hubs.length, w: Float64Array.from(x), n: 1, members: [{ item, phase }] };
            this.hubs.push(hub);
            return hub.id;
        }

        assimilate(id, x, item, phase) {
            const h = this.hubs[id];
            h.n += 1;
            const step = 1 / h.n; // running mean: shrinks as the hub matures
            for (let i = 0; i < this.dim; i++) h.w[i] += step * (x[i] - h.w[i]);
            h.members.push({ item, phase });
        }

        // Perceptual-mode trial: competition + vigilance over perceptual dims.
        perceive(item, engagement, phase) {
            const { x, cleared } = this.buildInput(item, engagement);
            const sims = this.matchAll(x);
            let winner = -1, best = -Infinity;
            sims.forEach((s, i) => { if (s > best) { best = s; winner = i; } });
            let action;
            if (winner >= 0 && best >= this.p.rho) {
                this.assimilate(winner, x, item, phase);
                action = 'snap';
            } else {
                const prev = winner;
                winner = this.recruit(x, item, phase);
                action = 'recruit';
                best = prev >= 0 ? best : 0;
            }
            return { item, engagement, phase, mode: 'perceive', action, winner, sim: best, sims, cleared, x };
        }

        // Consequence-mode trial: the disclosed outcome gates which hub assimilates.
        act(item, phase) {
            const { x, cleared } = this.buildInput(item, this.p.actEngagement, { disclose: true });
            const [outFeature, outPresent] = OUTCOMES[item];
            const compatible = [];
            this.hubs.forEach((h, i) => {
                const o = this.outcomeOf(h);
                if (o && o.feature === outFeature && o.present === outPresent) compatible.push(i);
            });
            const sims = this.matchAll(x);
            let action, winner, best;
            if (compatible.length > 0) {
                winner = compatible[0]; best = sims[compatible[0]];
                for (const i of compatible) if (sims[i] > best) { best = sims[i]; winner = i; }
                this.assimilate(winner, x, item, phase);
                action = 'snap';
            } else {
                winner = this.recruit(x, item, phase);
                best = 0;
                action = 'recruit';
            }
            return { item, engagement: this.p.actEngagement, phase, mode: 'act', action, winner, sim: best, sims, cleared, x, outcome: { feature: outFeature, present: outPresent } };
        }

        // Sight-only probe (no learning): which hub wins, plus its expectation.
        probe(item, engagement, goalVec) {
            const { x, cleared } = this.buildInput(item, engagement);
            let xin = x;
            if (goalVec) {
                xin = new Float64Array(this.dim);
                for (let i = 0; i < this.dim; i++) xin[i] = x[i] * goalVec[i];
            }
            let norm2 = 0;
            for (const i of this.perceptualIdx) norm2 += xin[i] * xin[i];
            const sims = norm2 < 1e-6 ? this.hubs.map(() => 0) : this.matchAll(xin);
            let winner = -1, best = -Infinity;
            if (norm2 >= 1e-6) sims.forEach((s, i) => { if (s > best) { best = s; winner = i; } });
            else best = 0; // the goal found nothing to amplify: no category matches
            return { item, engagement, mode: 'probe', goal: !!goalVec, winner, sim: best, sims, cleared, x, xGoal: goalVec ? xin : null };
        }

        // The goal vector for an outcome (e.g. fends-off): the outcome-built hub's
        // template over perceptual units — its affordance signature.
        goalVector(outFeature) {
            let src = null;
            this.hubs.forEach(h => {
                const o = this.outcomeOf(h);
                if (o && o.feature === outFeature && o.present) src = h;
            });
            if (!src) return null;
            // Lens over PRESENT units only, sharpened so faint particulars fall away
            // and the affordance signature dominates. Absent units carry no goal
            // relevance — the goal can only amplify a feature that is there.
            const g = new Float64Array(this.dim);
            let max = 0;
            for (let i = 0; i < this.nF; i++) {
                if (this.features[i].band === 'outcome') continue;
                max = Math.max(max, src.w[2 * i]);
            }
            if (max < 1e-9) return null;
            for (let i = 0; i < this.nF; i++) {
                if (this.features[i].band === 'outcome') continue;
                g[2 * i] = Math.pow(src.w[2 * i] / max, this.p.goalSharpness);
            }
            return g;
        }

        // Members grouped for reporting; hubs with >= 2 members count as categories.
        summary() {
            return this.hubs.map(h => {
                const counts = {};
                h.members.forEach(m => { counts[m.item] = (counts[m.item] || 0) + 1; });
                return { id: h.id, n: h.n, counts, outcome: this.outcomeOf(h) };
            });
        }
    }

    // ---------- trial streams (Section 4 simulations) ----------
    function shuffled(arr, rand) {
        const a = arr.slice();
        for (let i = a.length - 1; i > 0; i--) {
            const j = Math.floor(rand() * (i + 1));
            [a[i], a[j]] = [a[j], a[i]];
        }
        return a;
    }

    // Phase 1 (Sim 1a): 8 items x engagement levels x reps, one shuffled stream.
    function phase1Trials(p, rand) {
        const items = ['robin', 'canary', 'salmon', 'sunfish', 'oak', 'pine', 'rose', 'daisy'];
        const trials = [];
        for (const item of items)
            for (const e of p.engagementLevels)
                for (let r = 0; r < p.repsPerLevel; r++)
                    trials.push({ kind: 'perceive', item, engagement: e });
        return shuffled(trials, rand);
    }

    // Phase 2a (Sim 2a): see the berries (they merge), then eat them (they split).
    function phase2aTrials(p, rand) {
        const trials = [];
        for (let r = 0; r < 3; r++) {
            trials.push({ kind: 'perceive', item: 'berryA', engagement: p.actEngagement });
            trials.push({ kind: 'perceive', item: 'berryB', engagement: p.actEngagement });
        }
        const eat = [];
        for (let r = 0; r < 5; r++) { eat.push({ kind: 'act', item: 'berryA' }); eat.push({ kind: 'act', item: 'berryB' }); }
        return trials.concat(shuffled(eat, rand));
    }

    // Phase 2b (Sim 2b): see the objects, then wield them.
    function phase2bTrials(p, rand) {
        const trials = [];
        for (let r = 0; r < 2; r++)
            for (const item of ['stone', 'branch', 'bone'])
                trials.push({ kind: 'perceive', item, engagement: p.actEngagement });
        const wield = [];
        for (let r = 0; r < 3; r++)
            for (const item of ['stone', 'branch', 'bone'])
                wield.push({ kind: 'act', item });
        wield.push({ kind: 'act', item: 'daisy' }); // wielding a daisy fails
        return shuffled(trials, rand).concat(shuffled(wield, rand));
    }

    // Phase 3 (Sim 3): goal-off / goal-on probes. No learning.
    function phase3Trials(p) {
        const probes = [];
        for (const item of ['branch', 'stone', 'daisy']) {
            probes.push({ kind: 'probe', item, engagement: p.actEngagement, goal: false });
            probes.push({ kind: 'probe', item, engagement: p.actEngagement, goal: true });
        }
        return probes;
    }

    // Run every phase, recording one history entry (with hub snapshot) per trial.
    function runAll(params) {
        const model = new ESCFModel(params);
        const p = model.p;
        const streamRand = mulberry32(p.seed + 1);
        const phases = [
            { name: 'Perceptual (Sim 1a)', trials: phase1Trials(p, streamRand) },
            { name: 'Consequence: berries (Sim 2a)', trials: phase2aTrials(p, streamRand) },
            { name: 'Consequence: weapons (Sim 2b)', trials: phase2bTrials(p, streamRand) },
            { name: 'Goal (Sim 3)', trials: phase3Trials(p) },
        ];
        const history = [];
        phases.forEach((phase, phaseIdx) => {
            for (const t of phase.trials) {
                let rec;
                if (t.kind === 'perceive') rec = model.perceive(t.item, t.engagement, phaseIdx);
                else if (t.kind === 'act') rec = model.act(t.item, phaseIdx);
                else {
                    const g = t.goal ? model.goalVector('fends-off') : null;
                    rec = model.probe(t.item, t.engagement, g);
                    rec.phase = phaseIdx;
                }
                rec.phaseName = phase.name;
                rec.phaseIdx = phaseIdx;
                rec.hubSnapshot = model.hubs.map(h => ({ id: h.id, w: Array.from(h.w), n: h.n }));
                rec.summary = model.summary();
                history.push(rec);
            }
        });
        return { model, history, phases: phases.map((ph, i) => ({ name: ph.name, count: ph.trials.length, idx: i })) };
    }

    const ESCF = { FEATURES, ITEMS, OUTCOMES, DEFAULTS, ESCFModel, runAll, mulberry32, cosineOn };
    root.ESCF = ESCF;
    if (typeof module !== 'undefined' && module.exports) module.exports = ESCF;
})(typeof window !== 'undefined' ? window : globalThis);
