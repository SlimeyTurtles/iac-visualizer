// Action Graph — arithmetic state lattice with action-indexed Hebbian learning.
// Shared by public/action_graph.html (browser) and scripts/tune_action_graph.js (node).
//
// Every integer triple (A, B, C) in the domain is a node. Each node branches
// through a fixed vocabulary of arithmetic ACTIONS (X += Y, X -= Y, X ×D, X ÷D,
// X = 0 for every target variable X); an action that leaves the domain, needs a
// non-integer result, or maps the state to itself has no edge.
//
// IAC spreads symmetrically from the two clamped endpoints (current + goal) with
// decay and no competition. Learned weight does NOT live on the state→state
// edge: it lives on the ACTION, indexed by how many steps the edge's source is
// from the clamped current state. The edge (8,4,2) --A+=B--> (12,4,2) at depth 0
// and (3,1,2) --A+=B--> (4,1,2) at depth 0 share the same weight slot, so what
// hardens is "step 1 is A += B", not a particular pair of values.
(function (root) {
    'use strict';

    const VARS = ['A', 'B', 'C'];

    const DEFAULTS = {
        aMax: 12, bMax: 12, cMax: 6,   // domain: A,B in 0..12, C in 0..6
        dMax: 3,                        // scale factors 2..dMax (and their reciprocals)
        w0: 0.003,                      // base excitatory weight on every edge
        decay: 0.1,
        inh: 0,                         // within-step competition: each node is inhibited by inh × (sum of the other activations at its depth)
        steps: 30,                      // settle iterations
        eta: 0.05,                      // Hebbian rate (per-slot share units)
        wMax: 0.03,                     // cap on a learned action weight
        budget: 0.06,                   // cap on the sum of learned weights in one step row (0 = off)
        mode: 'step',                   // 'step' = W[step][action]; 'action' = W[action] pooled over steps
        credit: 'coact',                // 'coact' = every edge's co-activation; 'path' = only the walked plan's edges
        onlyOnSuccess: false,           // gate learning on the plan readout reaching the goal
        theta: 0.25,                    // Hebbian threshold: credit uses max(0, a - theta) at both ends
        kMax: 10,                       // step rows (depth >= kMax shares the last row)
        maxWalk: 20,                    // greedy readout step limit
    };

    function mulberry32(seed) {
        let t = seed >>> 0;
        return function () {
            t = (t + 0x6D2B79F5) >>> 0;
            let r = Math.imul(t ^ (t >>> 15), 1 | t);
            r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
            return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
        };
    }

    // The action vocabulary. Order within a target: add, add, sub, sub, scales, zero.
    function buildActions(dMax) {
        const actions = [];
        for (let x = 0; x < 3; x++) {
            const others = [0, 1, 2].filter(i => i !== x);
            for (const y of others) {
                actions.push({ label: `${VARS[x]} += ${VARS[y]}`, short: `${VARS[x]}+=${VARS[y]}`, kind: 'add', target: x,
                    apply: (s) => { const n = s.slice(); n[x] = s[x] + s[y]; return n; } });
            }
            for (const y of others) {
                actions.push({ label: `${VARS[x]} −= ${VARS[y]}`, short: `${VARS[x]}-=${VARS[y]}`, kind: 'sub', target: x,
                    apply: (s) => { const n = s.slice(); n[x] = s[x] - s[y]; return n; } });
            }
            for (let d = 2; d <= dMax; d++) {
                actions.push({ label: `${VARS[x]} ×${d}`, short: `${VARS[x]}×${d}`, kind: 'scale', target: x,
                    apply: (s) => { const n = s.slice(); n[x] = s[x] * d; return n; } });
                actions.push({ label: `${VARS[x]} ÷${d}`, short: `${VARS[x]}÷${d}`, kind: 'scale', target: x,
                    apply: (s) => { if (s[x] === 0 || s[x] % d !== 0) return null; const n = s.slice(); n[x] = s[x] / d; return n; } });
            }
            actions.push({ label: `${VARS[x]} = 0`, short: `${VARS[x]}=0`, kind: 'zero', target: x,
                apply: (s) => { const n = s.slice(); n[x] = 0; return n; } });
        }
        return actions;
    }

    class ActionGraphModel {
        constructor(opts = {}) {
            Object.assign(this, DEFAULTS, opts);
            this.actions = buildActions(this.dMax);
            this.nA = this.actions.length;
            this._buildGraph();
            this.W = new Float64Array(this.kMax * this.nA);
            this.episodes = 0;
        }

        // ---- graph ----

        idx(a, b, c) { return (a * (this.bMax + 1) + b) * (this.cMax + 1) + c; }
        inDomain(s) {
            return s[0] >= 0 && s[0] <= this.aMax && s[1] >= 0 && s[1] <= this.bMax && s[2] >= 0 && s[2] <= this.cMax;
        }
        label(i) { return this.states[i].join(', '); }

        _buildGraph() {
            const states = [];
            for (let a = 0; a <= this.aMax; a++)
                for (let b = 0; b <= this.bMax; b++)
                    for (let c = 0; c <= this.cMax; c++) states.push([a, b, c]);
            this.states = states;
            const n = this.n = states.length;

            const src = [], dst = [], act = [];
            const outPtr = new Int32Array(n + 1);
            for (let i = 0; i < n; i++) {
                outPtr[i] = src.length;
                const s = states[i];
                for (let j = 0; j < this.nA; j++) {
                    const t = this.actions[j].apply(s);
                    if (!t || !this.inDomain(t)) continue;
                    const k = this.idx(t[0], t[1], t[2]);
                    if (k === i) continue;              // identity move (e.g. += 0) is not an edge
                    src.push(i); dst.push(k); act.push(j);
                }
            }
            outPtr[n] = src.length;
            this.src = Int32Array.from(src);
            this.dst = Int32Array.from(dst);
            this.act = Int32Array.from(act);
            this.outPtr = outPtr;
            this.nE = src.length;

            // CSR over incoming edges, for the backward walk in BFS-from-goal queries.
            const inPtr = new Int32Array(n + 1);
            for (let e = 0; e < this.nE; e++) inPtr[this.dst[e] + 1]++;
            for (let i = 0; i < n; i++) inPtr[i + 1] += inPtr[i];
            const inEdge = new Int32Array(this.nE), fill = inPtr.slice(0, n);
            for (let e = 0; e < this.nE; e++) inEdge[fill[this.dst[e]]++] = e;
            this.inPtr = inPtr; this.inEdge = inEdge;
        }

        // Forward BFS depth of every node from the clamped current state (-1 = unreachable).
        depths(startIdx) {
            const d = new Int32Array(this.n).fill(-1);
            const q = [startIdx]; d[startIdx] = 0;
            for (let h = 0; h < q.length; h++) {
                const u = q[h];
                for (let e = this.outPtr[u]; e < this.outPtr[u + 1]; e++) {
                    const v = this.dst[e];
                    if (d[v] < 0) { d[v] = d[u] + 1; q.push(v); }
                }
            }
            return d;
        }

        // ---- weights ----

        slot(k, a) { return this.mode === 'step' ? Math.min(k, this.kMax - 1) * this.nA + a : a; }
        learned(k, a) { return this.W[this.slot(k, a)]; }
        edgeWeights(depth) {
            const w = new Float64Array(this.nE);
            for (let e = 0; e < this.nE; e++) {
                const k = depth[this.src[e]];
                w[e] = this.w0 + (k < 0 ? 0 : this.W[this.slot(k, this.act[e])]);
            }
            return w;
        }
        resetLearning() { this.W.fill(0); this.episodes = 0; }

        // ---- dynamics ----

        // Symmetric spread along every edge, decay toward zero, clamped endpoints held
        // at 1. Pass goalIdx = null to clamp only the current state (test mode).
        // Competition is per step number: nodes at the same depth from the current
        // state inhibit each other. Clamped endpoints receive but never send it —
        // otherwise the clamped goal suppresses exactly the states that feed it.
        settle(startIdx, goalIdx, opts = {}) {
            const depth = this.depths(startIdx);
            const wE = this.edgeWeights(depth);
            const n = this.n, a = new Float64Array(n), net = new Float64Array(n);
            a[startIdx] = 1; if (goalIdx != null) a[goalIdx] = 1;
            const history = opts.history ? [Float32Array.from(a)] : null;
            let maxDepth = 0;
            for (let i = 0; i < n; i++) if (depth[i] > maxDepth) maxDepth = depth[i];
            const layerSum = new Float64Array(maxDepth + 1);
            for (let t = 0; t < this.steps; t++) {
                net.fill(0);
                for (let e = 0; e < this.nE; e++) {
                    const w = wE[e], i = this.src[e], j = this.dst[e];
                    net[j] += w * a[i];
                    net[i] += w * a[j];
                }
                if (this.inh > 0) {
                    layerSum.fill(0);
                    for (let i = 0; i < n; i++) {
                        if (depth[i] < 0 || i === startIdx || i === goalIdx) continue;
                        layerSum[depth[i]] += a[i];
                    }
                    for (let i = 0; i < n; i++) {
                        if (depth[i] < 0) continue;
                        const own = (i === startIdx || i === goalIdx) ? 0 : a[i];
                        net[i] -= this.inh * (layerSum[depth[i]] - own);
                    }
                }
                for (let i = 0; i < n; i++) {
                    const x = net[i];
                    let v = a[i] + (x > 0 ? x * (1 - a[i]) : x * a[i]) - this.decay * a[i];
                    a[i] = v < 0 ? 0 : (v > 1 ? 1 : v);
                }
                a[startIdx] = 1; if (goalIdx != null) a[goalIdx] = 1;
                if (history) history.push(Float32Array.from(a));
            }
            return { acts: a, depth, wE, history };
        }

        // Hebbian credit goes to (step, action) slots, not to state pairs. Each edge's
        // co-activation a[src]·a[dst] is pooled into its source depth's row, turned
        // into a share of that row's total, and the row grows by eta·share, so every
        // step row learns at the same rate and the actions that carried the most
        // joint activation take the largest piece. Row sums are held under `budget`.
        hebbian(acts, depth) {
            const rows = this.mode === 'step' ? this.kMax : 1;
            const coact = new Float64Array(rows * this.nA);
            for (let e = 0; e < this.nE; e++) {
                const k = depth[this.src[e]];
                if (k < 0) continue;
                const ai = acts[this.src[e]] - this.theta, aj = acts[this.dst[e]] - this.theta;
                if (ai > 0 && aj > 0) coact[this.slot(k, this.act[e])] += ai * aj;
            }
            const rowDelta = [];
            for (let r = 0; r < rows; r++) {
                let total = 0;
                for (let a = 0; a < this.nA; a++) total += coact[r * this.nA + a];
                rowDelta.push(total);
                if (total < 1e-6) continue;
                let sum = 0;
                for (let a = 0; a < this.nA; a++) {
                    const s = r * this.nA + a;
                    let w = this.W[s] + this.eta * coact[s] / total;
                    if (w > this.wMax) w = this.wMax;
                    this.W[s] = w; sum += w;
                }
                if (this.budget > 0 && sum > this.budget) {
                    const f = this.budget / sum;
                    for (let a = 0; a < this.nA; a++) this.W[r * this.nA + a] *= f;
                }
            }
            this.episodes++;
            return rowDelta;
        }

        // Path credit: only the actions the plan readout actually took earn weight,
        // step t of the walk feeding slot (t, action_t) with the co-activation of its
        // two endpoints. A literal "the step from state 1 to state 2 was A += B".
        hebbianPath(acts, plan) {
            const rows = this.mode === 'step' ? this.kMax : 1;
            const touched = new Set();
            for (let t = 0; t < plan.actions.length; t++) {
                const s = this.slot(t, plan.actions[t]);
                const c = acts[plan.path[t]] * acts[plan.path[t + 1]];
                this.W[s] = Math.min(this.wMax, this.W[s] + this.eta * c);
                touched.add(this.mode === 'step' ? Math.min(t, this.kMax - 1) : 0);
            }
            if (this.budget > 0) {
                for (const r of touched) {
                    let sum = 0;
                    for (let a = 0; a < this.nA; a++) sum += this.W[r * this.nA + a];
                    if (sum > this.budget) { const f = this.budget / sum; for (let a = 0; a < this.nA; a++) this.W[r * this.nA + a] *= f; }
                }
            }
            this.episodes++;
            return Array.from({ length: rows }, (_, r) => touched.has(r) ? 1 : 0);
        }

        // ---- readouts ----

        // Plan readout: from the current state, step to the most active unvisited
        // out-neighbour until the goal is reached, a dead end, or the walk limit.
        greedyPath(acts, startIdx, goalIdx) {
            const path = [startIdx], actions = [], visited = new Set([startIdx]);
            let cur = startIdx;
            for (let t = 0; t < this.maxWalk; t++) {
                if (cur === goalIdx) return { outcome: 'success', path, actions };
                let best = -1, bestAct = -1;
                for (let e = this.outPtr[cur]; e < this.outPtr[cur + 1]; e++) {
                    const v = this.dst[e];
                    if (visited.has(v)) continue;
                    if (acts[v] > bestAct) { bestAct = acts[v]; best = e; }
                }
                if (best < 0) return { outcome: 'dead_end', path, actions };
                cur = this.dst[best];
                visited.add(cur); path.push(cur); actions.push(this.act[best]);
            }
            return { outcome: cur === goalIdx ? 'success' : 'too_long', path, actions };
        }

        // "Does activation connect them?" — the widest forward route from current to
        // goal, scored by its weakest node. Binary search on a threshold with BFS.
        bottleneckPath(acts, startIdx, goalIdx) {
            const reach = (theta) => {
                const prev = new Int32Array(this.n).fill(-2);
                const q = [startIdx]; prev[startIdx] = -1;
                for (let h = 0; h < q.length; h++) {
                    const u = q[h];
                    if (u === goalIdx) return prev;
                    for (let e = this.outPtr[u]; e < this.outPtr[u + 1]; e++) {
                        const v = this.dst[e];
                        if (prev[v] === -2 && acts[v] >= theta) { prev[v] = e; q.push(v); }
                    }
                }
                return null;
            };
            let lo = 0, hi = 1, prev = reach(0);
            if (!prev) return { bottleneck: 0, path: null, actions: null };
            for (let it = 0; it < 14; it++) {
                const mid = (lo + hi) / 2;
                const p = reach(mid);
                if (p) { lo = mid; prev = p; } else hi = mid;
            }
            const path = [], actions = [];
            for (let v = goalIdx; v !== startIdx; ) { const e = prev[v]; path.push(v); actions.push(this.act[e]); v = this.src[e]; }
            path.push(startIdx); path.reverse(); actions.reverse();
            return { bottleneck: lo, path, actions };
        }

        // Fewest-hops route (ground truth for efficiency).
        shortestPath(startIdx, goalIdx) {
            const prev = new Int32Array(this.n).fill(-2);
            const q = [startIdx]; prev[startIdx] = -1;
            for (let h = 0; h < q.length; h++) {
                const u = q[h];
                if (u === goalIdx) break;
                for (let e = this.outPtr[u]; e < this.outPtr[u + 1]; e++) {
                    const v = this.dst[e];
                    if (prev[v] === -2) { prev[v] = e; q.push(v); }
                }
            }
            if (prev[goalIdx] === -2) return null;
            const path = [], actions = [];
            for (let v = goalIdx; v !== startIdx; ) { const e = prev[v]; path.push(v); actions.push(this.act[e]); v = this.src[e]; }
            path.push(startIdx); path.reverse(); actions.reverse();
            return { path, actions, hops: actions.length };
        }

        // ---- episodes ----

        episode(start, goal, opts = {}) {
            const si = this.idx(...start), gi = this.idx(...goal);
            const { acts, depth, wE, history } = this.settle(si, gi, opts);
            const plan = this.greedyPath(acts, si, gi);
            const bridge = this.bottleneckPath(acts, si, gi);
            const optimal = this.shortestPath(si, gi);
            let active = 0;
            for (let i = 0; i < this.n; i++) if (acts[i] > 0.01) active++;
            let learnedRows = null;
            const gate = opts.onlyOnSuccess ?? this.onlyOnSuccess;
            if (opts.learn && (!gate || plan.outcome === 'success')) {
                learnedRows = this.credit === 'path' ? this.hebbianPath(acts, plan) : this.hebbian(acts, depth);
            }
            return { start, goal, si, gi, acts, depth, wE, history, plan, bridge, optimal, active, learnedRows,
                     success: plan.outcome === 'success', hops: plan.actions.length,
                     optHops: optimal ? optimal.hops : null };
        }

        // Test mode: only the current state is clamped. The learned (step, action)
        // weights steer the spread; the readout asks which goal-shaped node (x, 0, 0)
        // ends up most active, and the test passes if it is the correct one.
        test(start, goal, opts = {}) {
            const si = this.idx(...start), gi = this.idx(...goal);
            const { acts, depth, wE, history } = this.settle(si, null, opts);
            const candidates = [];
            for (let a = 0; a <= this.aMax; a++) {
                const i = this.idx(a, 0, 0);
                if (i !== si) candidates.push({ a, i, act: acts[i] });
            }
            candidates.sort((p, q) => q.act - p.act);
            const best = candidates[0];
            const rank = candidates.findIndex(c => c.i === gi) + 1;
            const pi = best ? best.i : -1;
            const bridge = pi >= 0 ? this.bottleneckPath(acts, si, pi) : { bottleneck: 0, path: null, actions: null };
            const goalBridge = this.bottleneckPath(acts, si, gi);
            const optimal = this.shortestPath(si, gi);
            let active = 0;
            for (let i = 0; i < this.n; i++) if (acts[i] > 0.01) active++;
            const success = pi === gi;
            const plan = bridge.path
                ? { outcome: success ? 'success' : 'wrong_goal', path: bridge.path, actions: bridge.actions }
                : { outcome: 'no_route', path: [si], actions: [] };
            return { mode: 'test', start, goal, si, gi, acts, depth, wE, history, plan, bridge: goalBridge, optimal, active,
                     predicted: pi >= 0 ? this.states[pi] : null, pi, predictedAct: best ? best.act : 0,
                     goalAct: acts[gi], rank, candidates, success, learnedRows: null,
                     hops: plan.actions.length, optHops: optimal ? optimal.hops : null };
        }

        evaluateTest(tasks) {
            const byHops = {};
            let ok = 0;
            const results = tasks.map(t => {
                const r = this.test(t.start, t.goal);
                const h = r.optHops == null ? 'unreachable' : r.optHops;
                byHops[h] = byHops[h] || { n: 0, ok: 0 };
                byHops[h].n++;
                if (r.success) { ok++; byHops[h].ok++; }
                return { start: t.start, goal: t.goal, success: r.success, predicted: r.predicted, rank: r.rank, optHops: r.optHops };
            });
            return { n: tasks.length, ok, byHops, results };
        }

        // Curriculum: (A, B, C) -> ((A + B) mod C, 0, 0).
        static modTask(a, b, c) { return { start: [a, b, c], goal: [(a + b) % c, 0, 0] }; }

        sampleTask(rng, range) {
            for (let tries = 0; tries < 1000; tries++) {
                const a = range.aMin + Math.floor(rng() * (range.aMax - range.aMin + 1));
                const b = range.aMin + Math.floor(rng() * (range.aMax - range.aMin + 1));
                const c = range.cMin + Math.floor(rng() * (range.cMax - range.cMin + 1));
                const t = ActionGraphModel.modTask(a, b, c);
                if (!this.inDomain(t.start) || !this.inDomain(t.goal)) continue;
                if (t.start.join() === t.goal.join()) continue;
                return t;
            }
            return null;
        }

        allTasks(range) {
            const out = [];
            for (let a = range.aMin; a <= range.aMax; a++)
                for (let b = range.aMin; b <= range.aMax; b++)
                    for (let c = range.cMin; c <= range.cMax; c++) {
                        const t = ActionGraphModel.modTask(a, b, c);
                        if (!this.inDomain(t.start) || !this.inDomain(t.goal)) continue;
                        if (t.start.join() === t.goal.join()) continue;
                        out.push(t);
                    }
            return out;
        }

        // Frozen-weights evaluation over a task list; success broken down by optimal hops.
        evaluate(tasks) {
            const byHops = {};
            let ok = 0, effSum = 0, effN = 0;
            const results = tasks.map(t => {
                const r = this.episode(t.start, t.goal, { learn: false });
                const h = r.optHops == null ? 'unreachable' : r.optHops;
                byHops[h] = byHops[h] || { n: 0, ok: 0 };
                byHops[h].n++;
                if (r.success) { ok++; byHops[h].ok++; effSum += r.optHops / r.hops; effN++; }
                return { start: t.start, goal: t.goal, success: r.success, hops: r.hops, optHops: r.optHops,
                         outcome: r.plan.outcome, bottleneck: r.bridge.bottleneck };
            });
            return { n: tasks.length, ok, byHops, meanEfficiency: effN ? effSum / effN : 0, results };
        }
    }

    const ActionGraph = { VARS, DEFAULTS, buildActions, ActionGraphModel, mulberry32 };
    root.ActionGraph = ActionGraph;
    if (typeof module !== 'undefined' && module.exports) module.exports = ActionGraph;
})(typeof window !== 'undefined' ? window : globalThis);
