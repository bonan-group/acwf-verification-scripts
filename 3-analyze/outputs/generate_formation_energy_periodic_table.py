#!/usr/bin/env python
"""
Generate interactive HTML periodic table viewer for formation energy comparisons.

Creates a self-contained HTML file displaying an interactive periodic table
where each element cell shows formation energy differences (meV/atom) between
two selected DFT codes, color-coded by magnitude. Supports raw and O-shifted
differences.

Usage:
  python generate_formation_energy_periodic_table.py \
      --codes "CASTEP@PW|C19MK2" "abacus_c19mk2" "VASP@PW|GW-PAW54*" \
      --output formation_energy_periodic_table.html
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import formation_energy_analysis as fea

# ===== CONFIGURATION =====

__version__ = "1.0.0"

# Periodic table layout: element -> (row, column)
PERIODIC_TABLE_LAYOUT = {
    'H': (1, 1), 'He': (1, 18),
    'Li': (2, 1), 'Be': (2, 2),
    'B': (2, 13), 'C': (2, 14), 'N': (2, 15), 'O': (2, 16), 'F': (2, 17), 'Ne': (2, 18),
    'Na': (3, 1), 'Mg': (3, 2),
    'Al': (3, 13), 'Si': (3, 14), 'P': (3, 15), 'S': (3, 16), 'Cl': (3, 17), 'Ar': (3, 18),
    'K': (4, 1), 'Ca': (4, 2),
    'Sc': (4, 3), 'Ti': (4, 4), 'V': (4, 5), 'Cr': (4, 6), 'Mn': (4, 7), 'Fe': (4, 8),
    'Co': (4, 9), 'Ni': (4, 10), 'Cu': (4, 11), 'Zn': (4, 12),
    'Ga': (4, 13), 'Ge': (4, 14), 'As': (4, 15), 'Se': (4, 16), 'Br': (4, 17), 'Kr': (4, 18),
    'Rb': (5, 1), 'Sr': (5, 2),
    'Y': (5, 3), 'Zr': (5, 4), 'Nb': (5, 5), 'Mo': (5, 6), 'Tc': (5, 7), 'Ru': (5, 8),
    'Rh': (5, 9), 'Pd': (5, 10), 'Ag': (5, 11), 'Cd': (5, 12),
    'In': (5, 13), 'Sn': (5, 14), 'Sb': (5, 15), 'Te': (5, 16), 'I': (5, 17), 'Xe': (5, 18),
    'Cs': (6, 1), 'Ba': (6, 2),
    'La': (6, 3),
    'Hf': (6, 4), 'Ta': (6, 5), 'W': (6, 6), 'Re': (6, 7), 'Os': (6, 8), 'Ir': (6, 9),
    'Pt': (6, 10), 'Au': (6, 11), 'Hg': (6, 12),
    'Tl': (6, 13), 'Pb': (6, 14), 'Bi': (6, 15), 'Po': (6, 16), 'At': (6, 17), 'Rn': (6, 18),
    'Fr': (7, 1), 'Ra': (7, 2),
    'Ac': (7, 3),
    'Rf': (7, 4), 'Db': (7, 5), 'Sg': (7, 6), 'Bh': (7, 7), 'Hs': (7, 8), 'Mt': (7, 9),
    'Ds': (7, 10), 'Rg': (7, 11), 'Cn': (7, 12),
    'Nh': (7, 13), 'Fl': (7, 14), 'Mc': (7, 15), 'Lv': (7, 16), 'Ts': (7, 17), 'Og': (7, 18),
    'Ce': (9, 4), 'Pr': (9, 5), 'Nd': (9, 6), 'Pm': (9, 7), 'Sm': (9, 8), 'Eu': (9, 9),
    'Gd': (9, 10), 'Tb': (9, 11), 'Dy': (9, 12), 'Ho': (9, 13), 'Er': (9, 14), 'Tm': (9, 15),
    'Yb': (9, 16), 'Lu': (9, 17),
    'Th': (10, 4), 'Pa': (10, 5), 'U': (10, 6), 'Np': (10, 7), 'Pu': (10, 8), 'Am': (10, 9),
    'Cm': (10, 10), 'Bk': (10, 11), 'Cf': (10, 12), 'Es': (10, 13), 'Fm': (10, 14), 'Md': (10, 15),
    'No': (10, 16), 'Lr': (10, 17),
}

DISABLED_ELEMENTS = [
    'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn',
    'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
]

OXIDE_CONFIGS = ['XO', 'XO2', 'XO3', 'X2O', 'X2O3', 'X2O5']


# ===== DATA PREPARATION =====

def prepare_formation_energy_data(codes):
    """Compute formation energy comparisons for all code pairs.

    Returns a dict structured for JavaScript consumption:
        {
            "codes": [list of code names],
            "comparisons": {
                "codeA|codeB": {
                    "delta_O": float (meV),
                    "compounds": {
                        "Element-Config": {
                            "eform_a": float (eV/atom),
                            "eform_b": float (eV/atom),
                            "raw_diff": float (meV/atom),
                            "shifted_diff": float (meV/atom)
                        }
                    }
                }
            }
        }
    """
    # Pre-compute formation energies for each code
    code_forms = {}
    for code in codes:
        print(f"  Computing formation energies: {code}")
        form, _, _ = fea.compute_formation_energies(code, per_atom=True)
        code_forms[code] = form

    data = {"codes": codes, "comparisons": {}}

    for i, code_a in enumerate(codes):
        for code_b in codes[i + 1:]:
            print(f"  Comparing: {code_a} vs {code_b}")
            form_a = code_forms[code_a]
            form_b = code_forms[code_b]

            common = sorted(set(form_a) & set(form_b))
            if not common:
                continue

            # Raw differences
            raw_diffs = {k: form_a[k] - form_b[k] for k in common}

            # Fit O shift (per-atom: weight = b/(a+b)), excluding outliers > 50 meV
            OUTLIER_THRESH = 0.050  # eV/atom
            weights = {}
            for k in common:
                a, b = fea.STOICH[k.split("-", 1)[1]]
                weights[k] = b / (a + b)
            sum_w_delta = sum(weights[k] * raw_diffs[k] for k in common
                              if abs(raw_diffs[k]) <= OUTLIER_THRESH)
            sum_w2 = sum(weights[k] ** 2 for k in common
                         if abs(raw_diffs[k]) <= OUTLIER_THRESH)
            delta_O = -sum_w_delta / sum_w2 if sum_w2 > 0 else 0.0

            # Shifted differences
            shifted_diffs = {}
            for k in common:
                shifted_diffs[k] = raw_diffs[k] + weights[k] * delta_O

            pair_key = f"{code_a}|{code_b}"
            pair_key_rev = f"{code_b}|{code_a}"

            compounds = {}
            for k in common:
                compounds[k] = {
                    "eform_a": form_a[k],
                    "eform_b": form_b[k],
                    "raw_diff": raw_diffs[k] * 1000,       # meV/atom
                    "shifted_diff": shifted_diffs[k] * 1000 # meV/atom
                }

            entry = {"delta_O": delta_O * 1000, "compounds": compounds}
            data["comparisons"][pair_key] = entry

            # Reverse pair: negate differences
            rev_compounds = {}
            for k in common:
                rev_compounds[k] = {
                    "eform_a": form_b[k],
                    "eform_b": form_a[k],
                    "raw_diff": -compounds[k]["raw_diff"],
                    "shifted_diff": -compounds[k]["shifted_diff"],
                }
            data["comparisons"][pair_key_rev] = {
                "delta_O": -delta_O * 1000,
                "compounds": rev_compounds,
            }

    return data


# ===== HTML GENERATION =====

def generate_periodic_table_html():
    """Generate HTML for the periodic table grid (oxides only)."""
    html = '<div class="periodic-table" id="periodic-table">\n'

    for element, (row, col) in PERIODIC_TABLE_LAYOUT.items():
        disabled_class = ' disabled' if element in DISABLED_ELEMENTS else ''
        html += f'  <div class="element-cell{disabled_class}" '
        html += f'data-element="{element}" '
        html += f'style="grid-column: {col}; grid-row: {row};" '
        html += f'onclick="onElementClick(\'{element}\')">\n'
        html += f'    <span class="element-symbol">{element}</span>\n'
        html += '    <div class="element-cell-inner oxides">\n'

        for idx, config in enumerate(OXIDE_CONFIGS):
            html += f'      <div class="config-subcell" data-config="{config}" data-index="{idx}"></div>\n'

        html += '    </div>\n'
        html += '  </div>\n'

    html += '</div>\n'
    return html


def get_css_styles():
    return '''
        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: #f0f8ff; color: #333; line-height: 1.6;
        }

        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }

        header {
            text-align: center; margin-bottom: 30px; padding: 20px;
            background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        header h1 { color: #0d6efd; font-size: 2em; margin-bottom: 10px; }
        .subtitle { color: #666; font-size: 1.1em; }

        .control-panel {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px; margin-bottom: 30px;
        }
        .control-section {
            background: white; padding: 20px; border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .control-section h3 { margin-bottom: 15px; color: #0d6efd; font-size: 1.2em; }

        select, .code-dropdown {
            width: 100%; padding: 10px; font-size: 1em;
            border: 1px solid #ddd; border-radius: 4px; background: white;
        }
        select:focus, .code-dropdown:focus {
            outline: none; border-color: #0d6efd;
            box-shadow: 0 0 0 3px rgba(13, 110, 253, 0.1);
        }

        .code-selector-grid {
            display: grid; grid-template-columns: 1fr auto 1fr;
            gap: 15px; align-items: center;
        }
        .code-dropdown-group { display: flex; flex-direction: column; gap: 8px; }
        .code-dropdown-group label { font-weight: 600; color: #666; font-size: 0.95em; }
        .vs-separator { font-size: 1.2em; font-weight: bold; color: #0d6efd; text-align: center; }

        .periodic-table-container { position: relative; margin: 30px 0; overflow-x: auto; }

        .periodic-table {
            display: grid; grid-template-columns: repeat(18, 70px);
            grid-template-rows: repeat(7, 65px) 15px repeat(2, 65px);
            gap: 2px; margin: 0 auto; width: fit-content; padding: 20px;
            background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .element-cell {
            position: relative; border: 1px solid #999; border-radius: 4px;
            background: white; cursor: pointer; user-select: none;
            transition: all 0.2s; height: 65px;
        }
        .element-cell:hover:not(.disabled) {
            transform: scale(1.05); box-shadow: 0 4px 8px rgba(0,0,0,0.2); z-index: 10;
        }
        .element-cell.disabled { background: #e0e0e0; cursor: not-allowed; opacity: 0.5; }

        .element-symbol {
            position: absolute; top: 1px; left: 3px; font-size: 10px;
            font-weight: bold; color: #333; z-index: 1; pointer-events: none;
        }

        .element-cell-inner { display: grid; width: 100%; height: 100%; padding-top: 12px; }
        .element-cell-inner.oxides { grid-template-columns: 1fr 1fr 1fr; grid-template-rows: 1fr 1fr; }

        .config-subcell {
            border: 0.5px solid #ddd; background: #fafafa;
            display: flex; align-items: center; justify-content: center;
            font-size: 10px; font-weight: 700; color: #000;
            text-shadow: 0 0 3px rgba(255,255,255,0.9); overflow: hidden;
        }
        .config-subcell.no-data { background: #f5f5f5; color: #999; }

        .colorbar-container {
            margin: 30px 0; padding: 20px; background: white;
            border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .colorbar-legend { display: flex; align-items: center; justify-content: center; gap: 20px; }
        .colorbar-gradient { width: 500px; height: 30px; border-radius: 4px; }
        .colorbar-labels {
            display: flex; justify-content: space-between; width: 500px;
            font-size: 0.9em; color: #666;
        }

        /* Detail panel */
        .detail-panel {
            margin: 30px 0; padding: 20px; background: white;
            border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .detail-panel-header {
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 20px; padding-bottom: 15px; border-bottom: 2px solid #0d6efd;
        }
        .detail-panel-header h3 { margin: 0; color: #0d6efd; }
        .close-button {
            padding: 8px 16px; background: #dc3545; color: white;
            border: none; border-radius: 4px; cursor: pointer;
        }
        .close-button:hover { background: #bb2d3b; }

        .detail-table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        .detail-table th {
            background: #0d6efd; color: white; padding: 10px 12px;
            text-align: left; font-size: 0.95em;
        }
        .detail-table td {
            padding: 8px 12px; border-bottom: 1px solid #dee2e6; font-size: 0.9em;
        }
        .detail-table tr:hover { background: #f8f9fa; }

        /* Summary stats */
        .stats-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px; margin: 20px 0;
        }
        .stat-card {
            background: #f8f9fa; padding: 15px; border-radius: 6px;
            text-align: center; border: 1px solid #dee2e6;
        }
        .stat-card .stat-value { font-size: 1.4em; font-weight: bold; color: #0d6efd; }
        .stat-card .stat-label { font-size: 0.85em; color: #666; margin-top: 5px; }

        footer { text-align: center; margin-top: 40px; padding: 20px; color: #666; font-size: 0.9em; }
    '''


def get_javascript_code(embedded_data):
    return f'''
        const DATA = {embedded_data};
        const CONFIGS = {json.dumps(OXIDE_CONFIGS)};

        const appState = {{
            codeA: null,
            codeB: null,
            metric: 'shifted',  // 'shifted' or 'raw'
            selectedElement: null,
            colorMax: 10,        // meV/atom — user-adjustable scale
            scatterChart: null,
            histChart: null
        }};

        // Color mapping for formation energy diffs (meV/atom)
        function hexToRgb(hex) {{
            const r = /^#?([a-f\\d]{{2}})([a-f\\d]{{2}})([a-f\\d]{{2}})$/i.exec(hex);
            return r ? {{ r: parseInt(r[1],16), g: parseInt(r[2],16), b: parseInt(r[3],16) }} : null;
        }}
        function rgbToHex(r,g,b) {{
            return "#" + ((1<<24)+(r<<16)+(g<<8)+b).toString(16).slice(1);
        }}
        function interpolateColor(c1, c2, f) {{
            const a = hexToRgb(c1), b = hexToRgb(c2);
            return rgbToHex(
                Math.round(a.r+(b.r-a.r)*f),
                Math.round(a.g+(b.g-a.g)*f),
                Math.round(a.b+(b.b-a.b)*f)
            );
        }}

        function getDiffColor(value) {{
            if (value === null || isNaN(value)) return '#fafafa';
            const v = Math.abs(value);
            const max = appState.colorMax;
            if (v >= max) return '#bf0000';                           // at or above max: dark red
            const frac = v / max;                                     // 0 to 1
            if (frac <= 0.10) return '#555998';                       // excellent: dark blue
            if (frac <= 0.50) return interpolateColor('#6B71AD','#EEE992', (frac-0.10)/0.40);
            return interpolateColor('#EEE992','#f53216', (frac-0.50)/0.50);
        }}

        function getPairKey() {{
            return appState.codeA + '|' + appState.codeB;
        }}

        function getDiff(element, config) {{
            const key = getPairKey();
            const pair = DATA.comparisons[key];
            if (!pair) return null;
            const compound = element + '-' + config;
            const c = pair.compounds[compound];
            if (!c) return null;
            return appState.metric === 'shifted' ? c.shifted_diff : c.raw_diff;
        }}

        function updatePeriodicTable() {{
            document.querySelectorAll('.element-cell:not(.disabled)').forEach(cell => {{
                const element = cell.dataset.element;
                cell.querySelectorAll('.config-subcell').forEach((subcell, idx) => {{
                    if (idx >= CONFIGS.length) return;
                    const config = CONFIGS[idx];
                    const diff = getDiff(element, config);
                    if (diff === null) {{
                        subcell.style.backgroundColor = '#fafafa';
                        subcell.classList.add('no-data');
                        subcell.textContent = '—';
                    }} else {{
                        subcell.style.backgroundColor = getDiffColor(diff);
                        subcell.classList.remove('no-data');
                        subcell.textContent = Math.abs(diff).toFixed(1);
                    }}
                }});
            }});
        }}

        function updateColorbar() {{
            const container = document.getElementById('colorbar-container');
            const pair = DATA.comparisons[getPairKey()];
            let statsHtml = '';
            if (pair) {{
                const diffs = Object.values(pair.compounds).map(c =>
                    appState.metric === 'shifted' ? c.shifted_diff : c.raw_diff
                );
                const absDiffs = diffs.map(d => Math.abs(d));
                const rms = Math.sqrt(diffs.reduce((s,d) => s+d*d, 0) / diffs.length);
                const mae = absDiffs.reduce((s,d) => s+d, 0) / absDiffs.length;
                const maxVal = Math.max(...absDiffs);
                const label = appState.metric === 'shifted' ? 'Shifted' : 'Raw';
                statsHtml = `
                    <h3 style="margin-bottom:15px;color:#0d6efd;">
                        ${{label}} diff:
                        <span style="color:#198754;">${{appState.codeA}}</span> vs
                        <span style="color:#198754;">${{appState.codeB}}</span>
                    </h3>
                    <div class="stats-grid">
                        <div class="stat-card"><div class="stat-value">${{diffs.length}}</div><div class="stat-label">Compounds</div></div>
                        <div class="stat-card"><div class="stat-value">${{rms.toFixed(1)}}</div><div class="stat-label">RMS (meV/atom)</div></div>
                        <div class="stat-card"><div class="stat-value">${{mae.toFixed(1)}}</div><div class="stat-label">MAE (meV/atom)</div></div>
                        <div class="stat-card"><div class="stat-value">${{maxVal.toFixed(1)}}</div><div class="stat-label">Max (meV/atom)</div></div>
                        <div class="stat-card"><div class="stat-value">${{pair.delta_O.toFixed(2)}}</div><div class="stat-label">O shift (meV)</div></div>
                    </div>`;
            }}

            container.innerHTML = statsHtml + `
                <div class="colorbar-legend" style="flex-wrap:wrap;">
                    <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                        <label for="color-max-input" style="font-weight:600;white-space:nowrap;">Color scale max (meV/atom):</label>
                        <input type="number" id="color-max-input" value="${{appState.colorMax}}" min="1" step="10"
                               style="width:80px;padding:5px 8px;font-size:1em;border:1px solid #ddd;border-radius:4px;">
                    </div>
                </div>
                <div class="colorbar-gradient" style="background:linear-gradient(to right,#555998,#6B71AD,#EEE992,#f53216,#bf0000);width:500px;height:30px;border-radius:4px;margin:0 auto;"></div>
                <div class="colorbar-labels" style="width:500px;margin:5px auto 0;">
                    <span>0</span>
                    <span>${{(appState.colorMax*0.10).toFixed(0)}}</span>
                    <span>${{(appState.colorMax*0.50).toFixed(0)}}</span>
                    <span>${{appState.colorMax.toFixed(0)}}+</span>
                </div>`;

            // Attach event listener to the new input
            document.getElementById('color-max-input').addEventListener('change', e => {{
                const val = parseFloat(e.target.value);
                if (val > 0) {{
                    appState.colorMax = val;
                    renderAll();
                }}
            }});
        }}

        function onElementClick(element) {{
            appState.selectedElement = element;
            showDetailPanel(element);
        }}

        function showDetailPanel(element) {{
            const panel = document.getElementById('detail-panel');
            const title = document.getElementById('detail-element-name');
            const tbody = document.getElementById('detail-tbody');

            title.textContent = element;
            tbody.innerHTML = '';

            const pair = DATA.comparisons[getPairKey()];
            if (!pair) {{ panel.style.display = 'none'; return; }}

            CONFIGS.forEach(config => {{
                const compound = element + '-' + config;
                const c = pair.compounds[compound];
                const tr = document.createElement('tr');

                if (c) {{
                    const diff = appState.metric === 'shifted' ? c.shifted_diff : c.raw_diff;
                    const color = getDiffColor(diff);
                    tr.innerHTML = `
                        <td>${{config}}</td>
                        <td>${{c.eform_a.toFixed(4)}}</td>
                        <td>${{c.eform_b.toFixed(4)}}</td>
                        <td style="color:${{diff > 0 ? '#dc3545' : '#198754'}}">${{c.raw_diff.toFixed(1)}}</td>
                        <td style="background:${{color}};color:#fff;font-weight:bold;">${{c.shifted_diff.toFixed(1)}}</td>`;
                }} else {{
                    tr.innerHTML = `<td>${{config}}</td><td colspan="4" style="color:#999;">No data</td>`;
                }}
                tbody.appendChild(tr);
            }});

            panel.style.display = 'block';
            setTimeout(() => panel.scrollIntoView({{ behavior: 'smooth', block: 'start' }}), 100);
        }}

        function closeDetailPanel() {{
            document.getElementById('detail-panel').style.display = 'none';
        }}

        function updateScatterPlot() {{
            const pair = DATA.comparisons[getPairKey()];
            document.getElementById('scatter-code-a').textContent = appState.codeA;
            document.getElementById('scatter-code-b').textContent = appState.codeB;
            if (!pair) {{
                if (appState.scatterChart) appState.scatterChart.destroy();
                if (appState.histChart) appState.histChart.destroy();
                return;
            }}

            const configColors = {{
                'XO': '#25cff2', 'XO2': '#3dd5f3', 'XO3': '#0d6efd',
                'X2O': '#198754', 'X2O3': '#ffc107', 'X2O5': '#dc3545'
            }};

            // Group by config for scatter coloring
            const byConfig = {{}};
            Object.entries(pair.compounds).forEach(([k, c]) => {{
                const config = k.split('-').slice(1).join('-');
                const diff = appState.metric === 'shifted' ? c.shifted_diff : c.raw_diff;
                if (!byConfig[config]) byConfig[config] = [];
                byConfig[config].push({{ x: c.eform_a, y: diff, label: k }});
            }});

            const scatterDatasets = Object.entries(byConfig).map(([config, pts]) => ({{
                label: config,
                data: pts,
                backgroundColor: configColors[config] || '#6c757d',
                pointRadius: 4,
                pointHoverRadius: 6,
            }}));

            // Axis range
            const allEform = Object.values(pair.compounds).map(c => c.eform_a);
            const allDiffs = Object.values(pair.compounds).map(c =>
                appState.metric === 'shifted' ? c.shifted_diff : c.raw_diff
            );
            const xmin = Math.min(...allEform);
            const xmax = Math.max(...allEform);
            const xpad = (xmax - xmin) * 0.05;
            const absDiffsForRange = allDiffs.map(d => Math.abs(d));
            const yMax = Math.max(appState.colorMax, Math.max(...absDiffsForRange)) * 1.1;

            // Scatter plot with zoom
            if (appState.scatterChart) appState.scatterChart.destroy();
            const ctx = document.getElementById('scatter-chart');
            appState.scatterChart = new Chart(ctx, {{
                type: 'scatter',
                data: {{ datasets: scatterDatasets }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: true,
                    aspectRatio: 1,
                    plugins: {{
                        legend: {{ position: 'top', labels: {{ usePointStyle: true }} }},
                        tooltip: {{
                            callbacks: {{
                                label: ctx => `${{ctx.raw.label}}: E_form=${{ctx.raw.x.toFixed(3)}}, diff=${{ctx.raw.y.toFixed(1)}} meV`
                            }}
                        }},
                        zoom: {{
                            pan: {{ enabled: true, mode: 'xy' }},
                            zoom: {{
                                wheel: {{ enabled: true }},
                                pinch: {{ enabled: true }},
                                mode: 'xy',
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            title: {{ display: true, text: `E_form ${{appState.codeA}} (eV/atom)` }},
                            min: xmin - xpad, max: xmax + xpad
                        }},
                        y: {{
                            title: {{ display: true, text: `Diff (meV/atom)` }},
                            suggestedMin: -yMax, suggestedMax: yMax
                        }}
                    }}
                }}
            }});

            // Histogram of errors
            const diffs = Object.values(pair.compounds).map(c =>
                appState.metric === 'shifted' ? c.shifted_diff : c.raw_diff
            );
            const absDiffs = diffs.map(d => Math.abs(d));
            const maxDiff = Math.max(...absDiffs);
            const histMax = Math.max(appState.colorMax, maxDiff);
            const nBins = 50;
            const binWidth = histMax / nBins;
            const bins = new Array(nBins).fill(0);
            absDiffs.forEach(v => {{
                const idx = Math.min(Math.floor(v / binWidth), nBins - 1);
                bins[idx]++;
            }});
            const binLabels = bins.map((_, i) => (i * binWidth).toFixed(1));

            if (appState.histChart) appState.histChart.destroy();
            const hctx = document.getElementById('histogram-chart');
            const metricLabel = appState.metric === 'shifted' ? 'Shifted' : 'Raw';
            appState.histChart = new Chart(hctx, {{
                type: 'bar',
                data: {{
                    labels: binLabels,
                    datasets: [{{
                        label: metricLabel + ' diff',
                        data: bins,
                        backgroundColor: bins.map((_, i) => getDiffColor(i * binWidth)),
                        borderWidth: 0,
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: true,
                    aspectRatio: 1,
                    plugins: {{
                        legend: {{ display: false }},
                    }},
                    scales: {{
                        x: {{
                            title: {{ display: true, text: '|Diff| (meV/atom)' }},
                            ticks: {{ maxTicksLimit: 10 }},
                        }},
                        y: {{
                            title: {{ display: true, text: 'Count' }},
                            beginAtZero: true,
                        }}
                    }}
                }}
            }});
        }}

        function renderAll() {{
            updatePeriodicTable();
            updateColorbar();
            updateScatterPlot();
            if (appState.selectedElement) showDetailPanel(appState.selectedElement);
        }}

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {{
            appState.codeA = document.getElementById('code-a-selector').value;
            appState.codeB = document.getElementById('code-b-selector').value;

            document.getElementById('code-a-selector').addEventListener('change', e => {{
                appState.codeA = e.target.value; renderAll();
            }});
            document.getElementById('code-b-selector').addEventListener('change', e => {{
                appState.codeB = e.target.value; renderAll();
            }});
            document.getElementById('metric-selector').addEventListener('change', e => {{
                appState.metric = e.target.value; renderAll();
            }});

            renderAll();
        }});
    '''


def generate_code_options_html(codes, default_index):
    html = ''
    for i, code in enumerate(codes):
        selected = ' selected' if i == default_index else ''
        html += f'<option value="{code}"{selected}>{code}</option>\n'
    return html


def generate_html(data, output_file):
    """Generate complete self-contained HTML file."""
    embedded_data = json.dumps(data, separators=(',', ':'))
    codes = data['codes']

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Formation Energy Comparison - Periodic Table</title>
    <style>
{get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Formation Energy Comparison</h1>
            <p class="subtitle">Per-atom oxide formation energy differences across DFT codes (meV/atom)</p>
        </header>

        <div class="control-panel">
            <div class="control-section">
                <h3>Select Codes</h3>
                <div class="code-selector-grid">
                    <div class="code-dropdown-group">
                        <label for="code-a-selector">Code A:</label>
                        <select id="code-a-selector" class="code-dropdown">
{generate_code_options_html(codes, 0)}
                        </select>
                    </div>
                    <div class="vs-separator">vs</div>
                    <div class="code-dropdown-group">
                        <label for="code-b-selector">Code B:</label>
                        <select id="code-b-selector" class="code-dropdown">
{generate_code_options_html(codes, min(1, len(codes)-1))}
                        </select>
                    </div>
                </div>
            </div>

            <div class="control-section">
                <h3>Metric</h3>
                <select id="metric-selector">
                    <option value="shifted">Shifted diff (after O-energy shift)</option>
                    <option value="raw">Raw diff (no O shift)</option>
                </select>
            </div>
        </div>

        <div class="periodic-table-container">
{generate_periodic_table_html()}
        </div>

        <div class="colorbar-container" id="colorbar-container"></div>

        <div class="scatter-panel" id="scatter-panel">
            <div class="scatter-panel-header">
                <h3>Formation Energy: <span id="scatter-code-a"></span> vs <span id="scatter-code-b"></span></h3>
            </div>
            <div style="display:flex;gap:20px;flex-wrap:wrap;justify-content:center;">
                <div style="position:relative;width:550px;max-width:100%;">
                    <canvas id="scatter-chart"></canvas>
                </div>
                <div style="position:relative;width:550px;max-width:100%;">
                    <canvas id="histogram-chart"></canvas>
                </div>
            </div>
        </div>

        <div class="detail-panel" id="detail-panel" style="display:none;">
            <div class="detail-panel-header">
                <h3>Element: <span id="detail-element-name"></span></h3>
                <button onclick="closeDetailPanel()" class="close-button">Close</button>
            </div>
            <table class="detail-table">
                <thead>
                    <tr>
                        <th>Config</th>
                        <th>E_form A (eV/atom)</th>
                        <th>E_form B (eV/atom)</th>
                        <th>Raw diff (meV/atom)</th>
                        <th>Shifted diff (meV/atom)</th>
                    </tr>
                </thead>
                <tbody id="detail-tbody"></tbody>
            </table>
        </div>

        <footer>
            <p>Generated by generate_formation_energy_periodic_table.py v{__version__}</p>
            <p>Data from ACWF verification project</p>
        </footer>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/hammerjs@2.0.8/hammer.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.0.1/dist/chartjs-plugin-zoom.min.js"></script>

    <script>
{get_javascript_code(embedded_data)}
    </script>
</body>
</html>'''

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Generated: {output_file}")


# ===== CLI =====

def main():
    parser = argparse.ArgumentParser(
        description='Generate interactive HTML periodic table for formation energy comparisons',
    )
    parser.add_argument('--codes', nargs='+', required=True,
                        help='Code names to include (must have both unaries + oxides data)')
    parser.add_argument('--output', default='formation_energy_periodic_table.html',
                        help='Output HTML filename')
    parser.add_argument('--results-dir', default='.',
                        help='Directory containing results JSON files (default: .)')
    args = parser.parse_args()

    # Set results directory
    fea.RESULTS_DIR = os.path.abspath(args.results_dir)

    # Validate codes
    available = fea.get_available_codes()
    valid_codes = []
    for code in args.codes:
        if code in available:
            valid_codes.append(code)
        else:
            print(f"WARNING: '{code}' not found or missing data, skipping", file=sys.stderr)

    if len(valid_codes) < 2:
        print("ERROR: Need at least 2 valid codes", file=sys.stderr)
        sys.exit(1)

    print(f"Preparing formation energy data for {len(valid_codes)} codes...")
    data = prepare_formation_energy_data(valid_codes)

    print(f"Generating HTML: {args.output}")
    generate_html(data, args.output)
    print("Done!")


if __name__ == '__main__':
    main()
