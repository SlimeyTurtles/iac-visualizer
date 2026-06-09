// Shared floating navigation menu, injected on every page.
// Provides a quick way back to the home page and to switch between models.
(function () {
    const PAGES = [
        { href: '/', label: 'Home', section: '' },
        { href: '/iac.html', label: 'IAC Visualizer', section: 'Supervised' },
        { href: '/rumelhart.html', label: 'Rumelhart Model', section: 'Supervised' },
        { href: '/unsupervised.html', label: 'Concept Learning (Default)', section: 'Unsupervised' },
        { href: '/unsupervised_csv.html', label: 'Concept Learning (CSV)', section: 'Unsupervised' },
        { href: '/sensory_dropout.html', label: 'Hierarchy Emergence ART', section: 'Unsupervised' },
        { href: '/synthetic_12_clean.html', label: 'Hierarchy Emergence (Synthetic 12, Clean)', section: 'Unsupervised' },
        { href: '/synthetic_12.html', label: 'Hierarchy Emergence (Synthetic 12, Dropout)', section: 'Unsupervised' },
        { href: '/synthetic_12_decay.html', label: 'Hierarchy Emergence (Corrupted + Decay)', section: 'Unsupervised' },
        { href: '/hierarchical_art.html', label: 'Forced Hierarchical ART', section: 'Unsupervised' },
    ];

    const style = document.createElement('style');
    style.textContent = `
        .nav-fab {
            position: fixed;
            top: 16px;
            right: 16px;
            z-index: 9999;
            width: 44px;
            height: 44px;
            border-radius: 22px;
            background: #111827;
            border: 1px solid #334155;
            color: #f1f5f9;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
            transition: all 0.15s;
            font-size: 18px;
        }
        .nav-fab:hover {
            background: #1e293b;
            border-color: #f59e0b;
            transform: translateY(-1px);
        }
        .nav-fab svg {
            width: 22px;
            height: 22px;
            stroke: currentColor;
        }
        .nav-panel {
            position: fixed;
            top: 70px;
            right: 16px;
            z-index: 9998;
            background: #111827;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 12px;
            min-width: 240px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
            display: none;
            max-height: calc(100vh - 100px);
            overflow-y: auto;
        }
        .nav-panel.open {
            display: block;
        }
        .nav-panel .nav-section {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #64748b;
            margin: 8px 8px 4px 8px;
            font-weight: 600;
        }
        .nav-panel a.nav-item {
            display: block;
            padding: 8px 10px;
            border-radius: 6px;
            color: #e2e8f0;
            text-decoration: none;
            font-size: 0.9rem;
            transition: all 0.15s;
        }
        .nav-panel a.nav-item:hover {
            background: #1e293b;
            color: #f59e0b;
        }
        .nav-panel a.nav-item.active {
            background: #1e293b;
            color: #f59e0b;
            font-weight: 600;
        }
        .nav-panel a.home-item {
            background: #1e293b;
            border: 1px solid #334155;
            margin-bottom: 8px;
            text-align: center;
            font-weight: 600;
        }
        .nav-panel a.home-item:hover {
            border-color: #f59e0b;
        }
        .nav-backdrop {
            position: fixed;
            inset: 0;
            background: transparent;
            z-index: 9997;
            display: none;
        }
        .nav-backdrop.open {
            display: block;
        }
    `;
    document.head.appendChild(style);

    const fab = document.createElement('button');
    fab.className = 'nav-fab';
    fab.title = 'Menu';
    fab.setAttribute('aria-label', 'Open menu');
    fab.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="3" y1="6" x2="21" y2="6"></line><line x1="3" y1="12" x2="21" y2="12"></line><line x1="3" y1="18" x2="21" y2="18"></line></svg>`;

    const backdrop = document.createElement('div');
    backdrop.className = 'nav-backdrop';

    const panel = document.createElement('div');
    panel.className = 'nav-panel';

    const currentPath = window.location.pathname.replace(/\/$/, '') || '/';

    const home = PAGES[0];
    const homeLink = document.createElement('a');
    homeLink.href = home.href;
    homeLink.className = 'nav-item home-item';
    homeLink.textContent = '← Home';
    panel.appendChild(homeLink);

    const grouped = {};
    PAGES.slice(1).forEach(p => {
        if (!grouped[p.section]) grouped[p.section] = [];
        grouped[p.section].push(p);
    });

    Object.entries(grouped).forEach(([section, items]) => {
        const label = document.createElement('div');
        label.className = 'nav-section';
        label.textContent = section;
        panel.appendChild(label);
        items.forEach(item => {
            const a = document.createElement('a');
            a.href = item.href;
            a.className = 'nav-item';
            a.textContent = item.label;
            if (currentPath === item.href || (item.href !== '/' && currentPath.endsWith(item.href))) {
                a.classList.add('active');
            }
            panel.appendChild(a);
        });
    });

    function setOpen(open) {
        panel.classList.toggle('open', open);
        backdrop.classList.toggle('open', open);
    }

    fab.addEventListener('click', (e) => {
        e.stopPropagation();
        setOpen(!panel.classList.contains('open'));
    });
    backdrop.addEventListener('click', () => setOpen(false));
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') setOpen(false);
    });

    function mount() {
        document.body.appendChild(backdrop);
        document.body.appendChild(panel);
        document.body.appendChild(fab);
    }

    // Number inputs marked with data-arrow-step accept any decimal (step="any"),
    // but Up/Down arrow keys and spinner buttons increment by data-arrow-step.
    function bindCustomStep() {
        function clamp(input, val) {
            const min = input.min !== '' ? parseFloat(input.min) : -Infinity;
            const max = input.max !== '' ? parseFloat(input.max) : Infinity;
            return Math.max(min, Math.min(max, val));
        }
        function step(input, dir) {
            const stepVal = parseFloat(input.dataset.arrowStep) || 1;
            const cur = parseFloat(input.value);
            const base = isNaN(cur) ? 0 : cur;
            // Snap to nearest multiple of step, then step by step
            const next = Math.round(base / stepVal) * stepVal + dir * stepVal;
            // Round to avoid floating point noise (e.g. 0.55 + 0.05 = 0.6000...01)
            const decimals = (input.dataset.arrowStep.split('.')[1] || '').length;
            const rounded = parseFloat(next.toFixed(decimals + 2));
            input.value = clamp(input, rounded);
            input.dispatchEvent(new Event('input', { bubbles: true }));
            input.dispatchEvent(new Event('change', { bubbles: true }));
        }
        document.addEventListener('keydown', (e) => {
            const t = e.target;
            if (!(t instanceof HTMLInputElement)) return;
            if (t.type !== 'number' || !t.dataset.arrowStep) return;
            if (e.key === 'ArrowUp') { e.preventDefault(); step(t, +1); }
            else if (e.key === 'ArrowDown') { e.preventDefault(); step(t, -1); }
        });
        // The native spinner buttons can't be hooked directly cross-browser.
        // Intercept the "step up/down" via input event when value snaps to wrong grid.
    }

    function init() {
        mount();
        bindCustomStep();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
