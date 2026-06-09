const express = require('express');
const { spawn } = require('child_process');
const path = require('path');

const app = express();
const port = 3000;

const PYTHON_DIR = path.join(__dirname, 'server-python');

app.use(express.json());
// Disable caching during development so HTML/CSS/JS changes are always fresh.
app.use((req, res, next) => {
    res.set('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate');
    res.set('Pragma', 'no-cache');
    res.set('Expires', '0');
    next();
});
app.use(express.static(path.join(__dirname, 'public')));

// Spawn a python bridge script in server-python/ and stream JSON back.
function runPythonBridge(scriptName, args, res) {
    const python = spawn('python3', [scriptName, ...args], { cwd: PYTHON_DIR });
    let dataString = '';
    let errorString = '';
    python.stdout.on('data', (data) => dataString += data.toString());
    python.stderr.on('data', (data) => errorString += data.toString());
    python.on('close', (code) => {
        if (code !== 0) {
            console.error(`Python error (${scriptName}):`, errorString);
            return res.status(500).json({ error: errorString });
        }
        try {
            res.json(JSON.parse(dataString));
        } catch (e) {
            res.status(500).json({ error: 'Failed to parse response', raw: dataString });
        }
    });
}

// ---- IAC endpoints (the only model that still uses the Python backend) ----
app.get('/network-info', (req, res) => {
    runPythonBridge('iac_bridge.py', ['info'], res);
});

app.post('/run-simulation', (req, res) => {
    const { selectedNodes, params } = req.body;
    runPythonBridge('iac_bridge.py', ['run', JSON.stringify(selectedNodes), JSON.stringify(params)], res);
});

app.listen(port, () => {
    console.log(`\nIAC Network Models running at http://localhost:${port}`);
    console.log(`\nAvailable models:`);
    console.log(`  - Home:                    http://localhost:${port}/`);
    console.log(`  - IAC Visualizer:          http://localhost:${port}/iac.html`);
    console.log(`  - Rumelhart:               http://localhost:${port}/rumelhart.html`);
    console.log(`  - Unsupervised:            http://localhost:${port}/unsupervised.html`);
    console.log(`  - Unsupervised CSV:        http://localhost:${port}/unsupervised_csv.html`);
    console.log(`  - Hierarchy Emergence ART: http://localhost:${port}/sensory_dropout.html`);
    console.log(`  - Hierarchy Emergence (Synthetic 12, Clean): http://localhost:${port}/synthetic_12_clean.html`);
    console.log(`  - Hierarchy Emergence (Synthetic 12, Dropout): http://localhost:${port}/synthetic_12.html`);
    console.log(`  - Hierarchy Emergence (Corrupted + Decay): http://localhost:${port}/synthetic_12_decay.html`);
    console.log(`  - Forced Hierarchical ART: http://localhost:${port}/hierarchical_art.html\n`);
});
