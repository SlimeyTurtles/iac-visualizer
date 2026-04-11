const express = require('express');
const { spawn } = require('child_process');
const path = require('path');

const app = express();
const port = 3000;

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Get initial network structure
app.get('/network-info', (req, res) => {
    const python = spawn('python3', ['iac_bridge.py', 'info']);

    let dataString = '';
    let errorString = '';

    python.stdout.on('data', (data) => dataString += data.toString());
    python.stderr.on('data', (data) => errorString += data.toString());

    python.on('close', (code) => {
        if (code !== 0) {
            console.error('Python error:', errorString);
            return res.status(500).json({ error: errorString });
        }
        try {
            res.json(JSON.parse(dataString));
        } catch (e) {
            res.status(500).json({ error: 'Failed to parse response', raw: dataString });
        }
    });
});

// Run the simulation
app.post('/run-simulation', (req, res) => {
    const { selectedNodes, params } = req.body;

    const python = spawn('python3', [
        'iac_bridge.py',
        'run',
        JSON.stringify(selectedNodes),
        JSON.stringify(params)
    ]);

    let dataString = '';
    let errorString = '';

    python.stdout.on('data', (data) => dataString += data.toString());
    python.stderr.on('data', (data) => errorString += data.toString());

    python.on('close', (code) => {
        if (code !== 0) {
            console.error('Python error:', errorString);
            return res.status(500).json({ error: errorString });
        }
        try {
            res.json(JSON.parse(dataString));
        } catch (e) {
            res.status(500).json({ error: 'Failed to parse response', raw: dataString });
        }
    });
});

// Rumelhart model endpoints
app.get('/rumelhart-info', (req, res) => {
    const params = req.query;
    const python = spawn('python3', ['rumelhart_bridge.py', 'info', JSON.stringify(params)]);

    let dataString = '';
    let errorString = '';

    python.stdout.on('data', (data) => dataString += data.toString());
    python.stderr.on('data', (data) => errorString += data.toString());

    python.on('close', (code) => {
        if (code !== 0) {
            console.error('Python error:', errorString);
            return res.status(500).json({ error: errorString });
        }
        try {
            res.json(JSON.parse(dataString));
        } catch (e) {
            res.status(500).json({ error: 'Failed to parse response', raw: dataString });
        }
    });
});

app.post('/rumelhart-train', (req, res) => {
    const { params } = req.body;

    const python = spawn('python3', [
        'rumelhart_bridge.py',
        'train',
        JSON.stringify(params)
    ]);

    let dataString = '';
    let errorString = '';

    python.stdout.on('data', (data) => dataString += data.toString());
    python.stderr.on('data', (data) => errorString += data.toString());

    python.on('close', (code) => {
        if (code !== 0) {
            console.error('Python error:', errorString);
            return res.status(500).json({ error: errorString });
        }
        try {
            res.json(JSON.parse(dataString));
        } catch (e) {
            res.status(500).json({ error: 'Failed to parse response', raw: dataString });
        }
    });
});

// Unsupervised model endpoints
app.get('/unsupervised-info', (req, res) => {
    const python = spawn('python3', ['unsupervised_bridge.py', 'info']);

    let dataString = '';
    let errorString = '';

    python.stdout.on('data', (data) => dataString += data.toString());
    python.stderr.on('data', (data) => errorString += data.toString());

    python.on('close', (code) => {
        if (code !== 0) {
            console.error('Python error:', errorString);
            return res.status(500).json({ error: errorString });
        }
        try {
            res.json(JSON.parse(dataString));
        } catch (e) {
            res.status(500).json({ error: 'Failed to parse response', raw: dataString });
        }
    });
});

app.post('/unsupervised-train', (req, res) => {
    const { params } = req.body;

    const python = spawn('python3', [
        'unsupervised_bridge.py',
        'train',
        JSON.stringify(params)
    ]);

    let dataString = '';
    let errorString = '';

    python.stdout.on('data', (data) => dataString += data.toString());
    python.stderr.on('data', (data) => errorString += data.toString());

    python.on('close', (code) => {
        if (code !== 0) {
            console.error('Python error:', errorString);
            return res.status(500).json({ error: errorString });
        }
        try {
            res.json(JSON.parse(dataString));
        } catch (e) {
            res.status(500).json({ error: 'Failed to parse response', raw: dataString });
        }
    });
});

app.listen(port, () => {
    console.log(`\nIAC Network Models running at http://localhost:${port}`);
    console.log(`\nAvailable models:`);
    console.log(`  - Home:              http://localhost:${port}/`);
    console.log(`  - IAC Visualizer:    http://localhost:${port}/iac.html`);
    console.log(`  - Rumelhart:         http://localhost:${port}/rumelhart.html`);
    console.log(`  - Unsupervised:      http://localhost:${port}/unsupervised.html`);
    console.log(`  - Unsupervised CSV:  http://localhost:${port}/unsupervised_csv.html\n`);
});
