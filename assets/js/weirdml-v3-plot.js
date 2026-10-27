// V3 chart presentation over precomputed running-best series.
// Data handling is unchanged: prepared curves, modes, scales and frontiers come from assets/data/weirdml_v3.json.
(() => {
const companyColors = {
            'OpenAI': '#74aa9c',
            'Anthropic': '#d97757',
            'Google': '#2ecc71',
            'DeepSeek': '#536dfe',
            'xAI': '#4d4d4d',
            'Meta': '#0668e1',
            'Mistral': '#ff7000',
            'Qwen': '#ff7f00',
            'Moonshot': '#0084ff',
            'ZhipuAI': '#10b981',
            'Gemma': '#2ecc71',
            'Inception': '#1a1a2e',
            'Nvidia': '#76b900',
            'MiniMax': '#ff4d6a',
            'Thinking Machines': '#161311',
            'Synthetic Strong': '#d97757', 'Synthetic Weak': '#74aa9c', 'Other': '#888888'
        };
const companies = {openai: 'OpenAI', anthropic: 'Anthropic', google: 'Google', deepseek: 'DeepSeek', 'z-ai': 'ZhipuAI', moonshotai: 'Moonshot', 'x-ai': 'xAI', meta: 'Meta', mistralai: 'Mistral', thinkingmachines: 'Thinking Machines'};
const icons = {OpenAI: 'openai_icon.png', Anthropic: 'claude_logo.png', Google: 'gemini_logo.png', DeepSeek: 'deepseek_whale.png', ZhipuAI: 'z-ai_logo.png', Moonshot: 'kimi_logo.png', xAI: 'grok_logo.png', Meta: 'meta_logo.png', Mistral: 'mistral_logo.png', 'Thinking Machines': 'thinking_machines_logo.png'};
const company = model => model.id === 'synthetic-strong' ? 'Synthetic Strong' : model.id === 'synthetic-weak' ? 'Synthetic Weak' : companies[model.slug.split('/')[0]] || 'Other';
const color = model => companyColors[company(model)];
const pngExport = new URLSearchParams(location.search).has('png');
const svg = d3.select('#chart');
const tooltip = d3.select('#tooltip');
let data, currentMode = 'overall', currentScale = 'log', currentGrid = 'configs', pinned = false, highlighted = null;
// Models picked in the legend stay highlighted across views; hover adds one more on top.
const selected = new Set();
const select = document.getElementById('task-select');
const openColor = '#1E90FF';
const closedColor = '#2E8B57';
const fmtPct = d3.format('.1%'), fmtUsd = d3.format('$.2f'), fmtTokens = d3.format('.3~s');
const isPhoneNow = () => window.innerWidth <= 600;
// Wording follows the input method: devices without hover get "tap".
const tapVerb = () => window.matchMedia('(hover: none), (pointer: coarse)').matches ? 'Tap' : 'Click';

// Rank within each lab by official overall score, keeping styles stable across tasks.
function curveDash(model) {
    const lab = model.slug.split('/')[0];
    const peers = data.models.filter(m => m.slug.split('/')[0] === lab)
        .sort((a, b) => b.score - a.score || a.id.localeCompare(b.id));
    const rank = peers.findIndex(m => m.id === model.id);
    const patterns = [null, '8 4', '2 4', '8 4 2 4', '8 4 2 4 2 4'];
    return rank < patterns.length ? patterns[rank] : `12 4 ${'2 4 '.repeat(rank - 2).trim()}`;
}
const harnessNames = {codex_cli: 'Codex CLI', claude_code: 'Claude Code', gemini_cli: 'Gemini CLI', opencode: 'OpenCode', kimi_code: 'Kimi Code'};
function harnessLabel(model) {
    const list = Array.isArray(model.harnesses) ? model.harnesses : [];
    const names = list.map(h => typeof h === 'string' ? (harnessNames[h] || h)
        : [harnessNames[h.name] || h.name, h.version ? 'v' + String(h.version).replace(/^v/, '') : ''].filter(Boolean).join(' '));
    return names.length ? names.join(', ') : null;
}

// ── Highlighting: hovered and legend-selected models in focus, the rest recede. ──
function setHighlight(id) {
    highlighted = id;
    // Open vs Closed has no model legend to show or clear a selection, so only hover applies there.
    const active = new Set(currentMode === 'frontier' ? [] : selected);
    if (id) active.add(id);
    const on = modelId => active.size > 0 && active.has(modelId);
    const off = modelId => active.size > 0 && !active.has(modelId);
    svg.selectAll('.series, .comparison-model')
        .classed('dimmed', d => off(d.model.id))
        .classed('focus', d => on(d.model.id));
    d3.select('#legend').selectAll('.legend-item')
        .classed('dimmed', d => Boolean(d) && off(d.id))
        .classed('focus', d => Boolean(d) && on(d.id))
        .classed('selected', d => Boolean(d) && selected.has(d.id))
        .attr('aria-pressed', d => d && !pngExport ? String(selected.has(d.id)) : null);
    d3.select('#legend').classed('has-selection', selected.size > 0).select('.legend-clear')
        .text(`Clear ${selected.size} selected`);
}
function toggleSelected(id) {
    if (selected.has(id)) selected.delete(id); else selected.add(id);
    setHighlight(highlighted);
}

// ── Model card: shown when a curve, marker or legend entry is hovered. ──
function modelCard(model, config) {
    tooltip.selectAll('*').remove();
    const head = tooltip.append('div').attr('class', 'tooltip-title');
    head.append('span').attr('class', 'tooltip-swatch').style('background', color(model));
    head.append('span').text(model.name);
    const rows = [];
    const configs = Array.isArray(config) ? config : config ? [config] : [];
    configs.forEach(entry => {
        const c = model.configurations[entry.id];
        rows.push([entry.name + (entry.hint_mode === 'Hints allowed' ? ' (hints)' : ''), null, 'section']);
        rows.push(['Task score', fmtPct(c.score)]);
        rows.push(['Final best', fmtPct(c.final_best)]);
        rows.push(['Runs', String(c.n)]);
        if (Number.isFinite(c.mean_api_cost_usd)) rows.push(['Cost / run', fmtUsd(c.mean_api_cost_usd)]);
    });
    if (configs.length) rows.push(['Overall', null, 'section']);
    rows.push(['Official score', fmtPct(model.score), 'strong']);
    if (Number.isFinite(model.mean_final_best)) rows.push(['Final best', fmtPct(model.mean_final_best)]);
    rows.push(['Cost / run', Number.isFinite(model.mean_api_cost_usd) ? fmtUsd(model.mean_api_cost_usd) : 'Unavailable']);
    rows.push(['Release date', model.release_date || 'Unknown']);
    rows.push(['Weights', model.open_weights == null ? 'Unknown' : model.open_weights ? 'Open' : 'Closed']);
    const harness = harnessLabel(model);
    if (harness) rows.push(['Harness', harness]);
    if (Number.isFinite(model.runs)) rows.push(['Valid runs', String(model.runs)]);
    rows.forEach(([label, value, kind]) => {
        if (kind === 'section') {tooltip.append('div').attr('class', 'tooltip-section').text(label); return;}
        const row = tooltip.append('div').attr('class', 'tooltip-row' + (kind === 'strong' ? ' strong' : ''));
        row.append('span').attr('class', 'tooltip-label').text(label);
        row.append('span').attr('class', 'tooltip-value').text(value);
    });
}
function placeTooltip(event) {
    if (isPhoneNow()) return;
    const node = tooltip.node();
    const w = node.offsetWidth || 300, h = node.offsetHeight || 200;
    let left = event.pageX + 18, top = event.pageY + 16;
    if (left + w > window.innerWidth - 12) left = Math.max(8, event.pageX - w - 18);
    // Stay beside the cursor and below it; near the bottom edge, slide up only as far as needed
    // (never flipping over the cursor), since the embed has no room below the legend.
    const bottom = window.innerHeight + window.scrollY - 12;
    if (top + h > bottom) top = Math.max(window.scrollY + 8, bottom - h);
    tooltip.style('left', left + 'px').style('top', top + 'px');
}
function dismiss() {
    pinned = false; hideTooltip(); setHighlight(null); svg.select('.guide').style('display', 'none');
}
function showTooltip(event) {
    // On phones the card is in flow beneath the chart; give it an explicit close control.
    if (isPhoneNow() && tooltip.select('.tooltip-close').empty()) {
        tooltip.select('.tooltip-title').append('button').attr('class', 'tooltip-close').attr('type', 'button')
            .attr('aria-label', 'Close').text('Close').on('click', event => {event.stopPropagation(); dismiss();});
    }
    tooltip.classed('visible', true);
    placeTooltip(event);
}
function hideTooltip() {
    tooltip.classed('visible', false);
}

// ── Markers: the lab icon at the end of each curve, or a coloured dot for labs without one. ──
const iconSize = (model, isPhone) => (pngExport ? 24 : isPhone ? 16 : 22) * (company(model) === 'OpenAI' ? 1.6 : 1);
// Half-width of the visible marker; the OpenAI image carries extra transparent padding.
const markerRadius = (model, isPhone) => icons[company(model)] ? iconSize(model, isPhone) / 2 * (company(model) === 'OpenAI' ? .75 : 1) : isPhone ? 5 : 8;
function drawMarker(parent, cx, cy, model, stroke, isPhone) {
    const icon = icons[company(model)];
    const marker = parent.append('g').attr('class', 'model-marker').attr('transform', `translate(${cx},${cy})`);
    if (icon) {
        const size = iconSize(model, isPhone);
        marker.append('image').attr('class', 'model-icon').attr('href', 'assets/icons/' + icon)
            .attr('x', -size / 2).attr('y', -size / 2).attr('width', size).attr('height', size);
    } else {
        marker.append('circle').attr('class', 'model-dot').attr('r', isPhone ? 5 : 8)
            .attr('fill', stroke).attr('stroke', '#fff').attr('stroke-width', 1.5);
    }
    return marker;
}
// Spread end labels apart vertically so nearby finishes stay legible. An item's `extra` is the added half-height of
// a wrapped label. When the labels cannot all fit, the gaps shrink evenly instead of spilling past the bounds.
function spreadLabels(items, minGap, lower, upper) {
    items.sort((a, b) => a.y - b.y);
    const gaps = items.slice(1).map((b, i) => minGap + (items[i].extra || 0) + (b.extra || 0));
    const needed = d3.sum(gaps), room = upper - lower;
    if (needed > room && needed > 0) gaps.forEach((v, i) => {gaps[i] = v * room / needed;});
    for (let i = 1; i < items.length; i++) if (items[i].y - items[i - 1].y < gaps[i - 1]) items[i].y = items[i - 1].y + gaps[i - 1];
    for (let i = items.length - 1; i >= 0; i--) {
        const cap = i === items.length - 1 ? upper : items[i + 1].y - gaps[i];
        if (items[i].y > cap) items[i].y = cap;
    }
    for (let i = 0; i < items.length; i++) {
        const floor = i === 0 ? lower : items[i - 1].y + gaps[i - 1];
        if (items[i].y < floor) items[i].y = floor;
    }
}

// Scatter labels go beside their own marker where they can, and otherwise as close as space allows, with a leader
// line. Each label gets candidate spots (hugging its marker, or anywhere nearby clear of markers); a seeded annealing
// search then picks one per label, minimising leader length plus penalties for overlaps, leaders crossing markers,
// labels or other leaders, and labels sitting nearer another model's marker than their own. Deterministic per layout.
function placeScatterLabels(items, bounds, gap) {
    const overlapArea = (a, b) => Math.max(0, Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0)) * Math.max(0, Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0));
    const box = (cx, cy, w, h) => ({x0: cx - w / 2, x1: cx + w / 2, y0: cy - h / 2, y1: cy + h / 2});
    const inside = b => b.x0 >= bounds.x0 && b.x1 <= bounds.x1 && b.y0 >= bounds.y0 && b.y1 <= bounds.y1;
    const nearest = (item, b) => [Math.min(Math.max(item.x, b.x0), b.x1), Math.min(Math.max(item.y, b.y0), b.y1)];
    // Liang–Barsky: does the segment from p to q pass through box b?
    const crossesBox = (p, q, b) => {
        let t0 = 0, t1 = 1;
        const dx = q[0] - p[0], dy = q[1] - p[1];
        const clip = (a, c) => {
            if (a === 0) return c >= 0;
            const t = c / a;
            if (a < 0) {if (t > t1) return false; if (t > t0) t0 = t;} else {if (t < t0) return false; if (t < t1) t1 = t;}
            return true;
        };
        return clip(-dx, p[0] - b.x0) && clip(dx, b.x1 - p[0]) && clip(-dy, p[1] - b.y0) && clip(dy, b.y1 - p[1]) && t0 < t1;
    };
    const side = (p, q, r) => Math.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]));
    const segmentsCross = (a, b) => side(a[0], a[1], b[0]) * side(a[0], a[1], b[1]) < 0 && side(b[0], b[1], a[0]) * side(b[0], b[1], a[1]) < 0;
    const markers = items.map(i => ({x0: i.x - i.r, x1: i.x + i.r, y0: i.y - i.r, y1: i.y + i.r}));
    const markerClash = b => markers.reduce((sum, m) => sum + overlapArea(b, m), 0);
    const reachLimit = Math.max(80, .45 * Math.min(bounds.x1 - bounds.x0, bounds.y1 - bounds.y0) + 60);
    const directions = [0, 180, -90, 90, -35, -145, 35, 145].map(d => d * Math.PI / 180);
    // Candidate spots, each with the cost that does not depend on other labels.
    const candidates = items.map((item, index) => {
        const {x, y, r, w, h} = item;
        const own = [x, y];
        const make = (b, near, rank) => {
            const end = nearest(item, b);
            const leader = near ? null : [own, end];
            const length = near ? rank * .5 : Math.hypot(end[0] - x, end[1] - y) + 8;
            const clash = markerClash(b);
            let cost = length + (clash > 0 ? 400 + clash * 4 : 0), misread = false;
            items.forEach((other, j) => {
                if (j === index) return;
                if (leader && crossesBox(own, end, markers[j])) cost += 40;
                const [ox, oy] = nearest(other, b);
                if (Math.hypot(ox - other.x, oy - other.y) + 2 < (near ? r + gap : Math.hypot(end[0] - x, end[1] - y))) misread = true;
            });
            // Read as belonging to a nearer marker: worse than a long leader.
            if (misread) cost += 150;
            return {b, leader, cost};
        };
        const list = [];
        directions.forEach((angle, rank) => {
            const cos = Math.cos(angle), sin = Math.sin(angle);
            // Push the box out along the direction until its nearest edge clears the marker by the gap.
            const half = Math.min(Math.abs(cos) > 1e-9 ? w / 2 / Math.abs(cos) : Infinity, Math.abs(sin) > 1e-9 ? h / 2 / Math.abs(sin) : Infinity);
            const b = box(x + cos * (r + gap + half), y + sin * (r + gap + half), w, h);
            if (inside(b)) list.push(make(b, true, rank));
        });
        // Grid spots clear of every marker, thinned to the nearest per coarse cell so they spread out and stay cheap.
        const cells = new Map();
        const top = bounds.y0 + h / 2 + Math.max(0, Math.floor((y - reachLimit - h - bounds.y0) / 5)) * 5;
        const left = bounds.x0 + w / 2 + Math.max(0, Math.floor((x - reachLimit - w - bounds.x0) / 6)) * 6;
        for (let cy = top; cy <= Math.min(bounds.y1 - h / 2, y + reachLimit + h); cy += 5) {
            for (let cx = left; cx <= Math.min(bounds.x1 - w / 2, x + reachLimit + w); cx += 6) {
                const dx = Math.max(cx - w / 2 - x, 0, x - cx - w / 2), dy = Math.max(cy - h / 2 - y, 0, y - cy - h / 2);
                const length = Math.hypot(dx, dy);
                if (length < r + gap * 2 || length > reachLimit) continue;
                const key = Math.floor((cx - w / 2) / 24) * 4096 + Math.floor((cy - h / 2) / 12);
                if ((cells.get(key)?.length ?? Infinity) <= length) continue;
                const b = box(cx, cy, w, h);
                if (markerClash(b) === 0) cells.set(key, {b, length});
            }
        }
        const closest = [...cells.values()].sort((a, b) => a.length - b.length).slice(0, 400);
        list.push(...closest.map(c => make(c.b, false, 0)).sort((a, b) => a.cost - b.cost).slice(0, 220));
        // Nothing fits (a tiny chart): fall back to the plain spot beside the marker.
        if (!list.length) list.push(make(box(Math.max(bounds.x0 + w / 2, x + r + gap + w / 2), y, w, h), true, 0));
        return list;
    });
    const pairCost = (a, b) => {
        const area = overlapArea(a.b, b.b);
        let cost = area > 0 ? 400 + area * 4 : 0;
        if (a.leader && crossesBox(a.leader[0], a.leader[1], b.b)) cost += 60;
        if (b.leader && crossesBox(b.leader[0], b.leader[1], a.b)) cost += 60;
        if (a.leader && b.leader && segmentsCross(a.leader, b.leader)) cost += 25;
        return cost;
    };
    const n = items.length;
    const choice = new Array(n).fill(0);
    const labelCost = (i, c) => {
        let cost = c.cost;
        for (let j = 0; j < n; j++) if (j !== i) cost += pairCost(c, candidates[j][choice[j]]);
        return cost;
    };
    // Start greedily, best scores first, then anneal and finish with a best-response sweep.
    const placedOrder = items.map((item, i) => i).sort((a, b) => items[a].priority - items[b].priority);
    const placed = new Set();
    placedOrder.forEach(i => {
        let best = 0, bestCost = Infinity;
        candidates[i].forEach((c, k) => {
            let cost = c.cost;
            placed.forEach(j => {cost += pairCost(c, candidates[j][choice[j]]);});
            if (cost < bestCost) {bestCost = cost; best = k;}
        });
        choice[i] = best; placed.add(i);
    });
    let seed = 20261014;
    const random = () => {seed = (seed + 0x6D2B79F5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296;};
    const steps = 300 * n;
    for (let step = 0; step < steps; step++) {
        const temperature = 60 * Math.pow(.5 / 60, step / steps);
        const i = Math.floor(random() * n), list = candidates[i];
        if (list.length < 2) continue;
        const k = Math.floor(random() * list.length);
        const delta = labelCost(i, list[k]) - labelCost(i, list[choice[i]]);
        if (delta < 0 || random() < Math.exp(-delta / temperature)) choice[i] = k;
    }
    for (let sweep = 0, changed = true; sweep < 10 && changed; sweep++) {
        changed = false;
        for (let i = 0; i < n; i++) {
            let best = choice[i], bestCost = labelCost(i, candidates[i][best]);
            candidates[i].forEach((c, k) => {const cost = labelCost(i, c); if (cost < bestCost - 1e-9) {bestCost = cost; best = k;}});
            if (best !== choice[i]) {choice[i] = best; changed = true;}
        }
    }
    items.forEach((item, i) => {const c = candidates[i][choice[i]]; item.box = c.b; item.leader = Boolean(c.leader);});
    return items;
}

// PNG exports draw the legend inside the SVG so the rasteriser sees one consistent image.
function legendEntries() {
    if (currentMode === 'frontier') return [
        {label: 'Closed Frontier', color: closedColor}, {label: 'Open Frontier', color: openColor},
        {label: 'Closed Model', color: closedColor, dot: true}, {label: 'Open Model', color: openColor, dot: true}];
    const lines = currentMode === 'overall' || currentMode === 'task';
    return data.models.map(m => ({label: m.name, color: color(m), dash: lines ? curveDash(m) : null, dot: !lines}));
}
function measureSvgLegend(width) {
    const probe = svg.append('text').attr('class', 'svg-legend-text').style('visibility', 'hidden');
    let x = 0, rows = 1;
    legendEntries().forEach(e => {
        probe.text(e.label);
        const w = 38 + probe.node().getComputedTextLength();
        if (x + w > width && x > 0) {x = 0; rows++;}
        x += w + 26;
    });
    probe.text('htihle.github.io/weirdml');
    if (x + probe.node().getComputedTextLength() > width) rows++;
    probe.remove();
    return rows;
}
function drawSvgLegend(g, width, y) {
    const legend = g.append('g').attr('class', 'svg-legend').attr('transform', `translate(0,${y})`);
    const rowH = 20;
    let x = 0, row = 0;
    legendEntries().forEach(e => {
        const item = legend.append('g');
        const text = item.append('text').attr('class', 'svg-legend-text').attr('x', 38).attr('dominant-baseline', 'middle').text(e.label);
        const w = 38 + text.node().getComputedTextLength();
        if (x + w > width && x > 0) {x = 0; row++;}
        item.attr('transform', `translate(${x},${row * rowH})`);
        if (e.dot) item.append('circle').attr('cx', 15).attr('cy', 0).attr('r', 5).attr('fill', e.color);
        else item.append('line').attr('x1', 0).attr('x2', 30).attr('y1', 0).attr('y2', 0)
            .attr('stroke', e.color).attr('stroke-width', 3).attr('stroke-dasharray', e.dash || null);
        x += w + 26;
    });
    const credit = legend.append('text').attr('class', 'svg-attribution').attr('x', width).attr('y', row * rowH)
        .attr('text-anchor', 'end').attr('dominant-baseline', 'middle').text('htihle.github.io/weirdml');
    if (x + credit.node().getComputedTextLength() > width) credit.attr('y', (row + 1) * rowH);
}
function updateLegend() {
    const legend = d3.select('#legend');
    legend.selectAll('*').remove();
    if (currentMode === 'grid') return;
    if (currentMode === 'frontier') {
        const row = legend.append('div').attr('class', 'legend-items static');
        [{label: 'Closed Frontier', color: closedColor}, {label: 'Open Frontier', color: openColor},
         {label: 'Closed Model', color: closedColor, dot: true}, {label: 'Open Model', color: openColor, dot: true}].forEach(entry => {
            const item = row.append('div').attr('class', 'legend-item').datum(null);
            if (entry.dot) item.append('span').attr('class', 'legend-dot').style('background', entry.color);
            else item.append('svg').attr('width', 24).attr('height', 14).append('line')
                .attr('x1', 0).attr('x2', 24).attr('y1', 7).attr('y2', 7).attr('stroke', entry.color).attr('stroke-width', 3);
            item.append('span').text(entry.label);
        });
        return;
    }
    if (!pngExport) {
        const head = legend.append('div').attr('class', 'legend-head');
        head.append('span').attr('class', 'legend-title').text('Models');
        head.append('span').attr('class', 'legend-hint')
            .text(`${tapVerb()} to highlight` + (isPhoneNow() ? '' : ' · select several to compare'));
        head.append('button').attr('type', 'button').attr('class', 'legend-clear')
            .on('click', () => {selected.clear(); dismiss();});
    }
    const row = legend.append('div').attr('class', 'legend-items').attr('role', pngExport ? null : 'group')
        .attr('aria-label', pngExport ? null : 'Highlight models');
    const lines = currentMode === 'overall' || currentMode === 'task';
    data.models.forEach(model => {
        const item = row.append(pngExport ? 'div' : 'button').attr('class', 'legend-item').datum(model)
            .attr('type', pngExport ? null : 'button').style('--c', color(model));
        const icon = icons[company(model)];
        if (icon) item.append('img').attr('class', 'legend-icon').attr('src', 'assets/icons/' + icon).attr('alt', '');
        else item.append('span').attr('class', 'legend-dot').style('background', color(model));
        if (lines) {
            item.append('svg').attr('class', 'legend-line').attr('width', 26).attr('height', 12).append('line')
                .attr('x1', 1).attr('x2', 25).attr('y1', 6).attr('y2', 6)
                .attr('stroke', color(model)).attr('stroke-width', 3)
                .attr('stroke-dasharray', curveDash(model));
        }
        item.append('span').attr('class', 'legend-name').text(model.name);
        if (pngExport) return;
        item.on('pointerenter', event => {if (!pinned && event.pointerType !== 'touch') {setHighlight(model.id); modelCard(model, currentConfig()); showTooltip(event);}})
            .on('pointermove', event => {if (!pinned) placeTooltip(event);})
            .on('pointerleave', () => {if (!pinned) {setHighlight(null); hideTooltip();}})
            // Clicking toggles a persistent highlight; several models can be selected at once.
            .on('click', event => selectFromClick(model, event))
            .on('focus', () => {if (!pinned) setHighlight(model.id);})
            .on('blur', () => {if (!pinned) setHighlight(null);});
    });
}
// A click (or tap) on a model anywhere toggles its selection. Touch has no hover,
// so a tap that selects also pins the model's card; deselecting closes it.
function selectFromClick(model, event, config = currentConfig()) {
    toggleSelected(model.id);
    if (event.pointerType === 'touch' || isPhoneNow()) {
        if (selected.has(model.id)) {pinned = true; setHighlight(model.id); modelCard(model, config); showTooltip(event);}
        else dismiss();
    }
}
const currentConfig = () => currentMode === 'task' ? data.configurations.find(c => c.id === select.value) : null;

function updateComparison() {
    const dateMode = currentMode === 'date' || currentMode === 'frontier';
    const isPhone = isPhoneNow();
    const container = svg.node().parentElement;
    const width = container.clientWidth - 20, height = Math.max(isPhone ? 180 : 200, container.clientHeight - 20);
    const legendRows = pngExport ? measureSvgLegend(width - 65 - 10) : 0;
    const margin = pngExport ? {top: 16, right: 28, bottom: 74 + (legendRows - 1) * 20, left: 65} : isPhone ? {top: 20, right: 20, bottom: 40, left: 45} : {top: 24, right: 40, bottom: 60, left: 65};
    const innerWidth = width - margin.left - margin.right, innerHeight = height - margin.top - margin.bottom;
    svg.attr('width', width).attr('height', height).on('pointermove', null).on('pointerleave', null).on('click', null);
    svg.selectAll('*').remove();
    tooltip.classed('mobile-panel', isPhone).classed('visible', false);
    pinned = false; highlighted = null;
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
    const models = data.models.filter(m => dateMode ? m.release_date : Number.isFinite(m.mean_api_cost_usd) && (currentScale !== 'log' || m.mean_api_cost_usd > 0));
    if (!models.length) {g.append('text').attr('x', 10).attr('y', 30).text(dateMode ? 'No release dates available.' : 'No positive API costs available.'); return;}
    const getX = model => dateMode ? new Date(model.release_date + 'T00:00:00Z') : model.mean_api_cost_usd;
    let xScale;
    if (dateMode) {
        let [start, end] = d3.extent(models, getX);
        if (currentMode === 'frontier') end = new Date(data.date_frontiers.as_of + 'T00:00:00Z');
        if (+start === +end) {start = new Date(+start - 86400000); end = new Date(+end + 86400000);}
        xScale = d3.scaleUtc().domain([start, end]).range([0, innerWidth]).nice();
    } else {
        let domain = currentScale === 'log' ? d3.extent(models, getX) : [0, d3.max(models, getX) * 1.05 || 1];
        if (domain[0] === domain[1]) domain = [domain[0] / 2, domain[1] * 2];
        xScale = (currentScale === 'log' ? d3.scaleLog() : d3.scaleLinear()).domain(domain).range([0, innerWidth]).nice();
    }
    const yScale = d3.scaleLinear().domain([0, (d3.max(models, m => m.score) || 1) * 1.05]).range([innerHeight, 0]).nice();
    const ticks = isPhone ? 4 : 6, fontSize = isPhone ? '9px' : pngExport ? '13px' : '12px';
    const makeAxis = factory => dateMode ? factory(xScale).ticks(ticks)
        : currentScale === 'log' ? factory(xScale).ticks(ticks, '.4~f') : factory(xScale).ticks(ticks);
    g.append('g').attr('class', 'grid').call(d3.axisLeft(yScale).tickSize(-innerWidth).tickFormat(''));
    g.append('g').attr('class', 'grid').call(makeAxis(d3.axisBottom).tickSize(innerHeight).tickFormat(''));
    g.append('g').attr('class', 'axis axis-x').attr('transform', `translate(0,${innerHeight})`).call(makeAxis(d3.axisBottom)).selectAll('text').style('font-size', fontSize);
    g.append('g').attr('class', 'axis axis-y').call(d3.axisLeft(yScale).tickFormat(d3.format('.0%')).ticks(isPhone ? 5 : null)).selectAll('text').style('font-size', fontSize);
    g.append('text').attr('class', 'axis-label').attr('x', innerWidth / 2).attr('y', innerHeight + (isPhone ? 30 : 45))
        .attr('text-anchor', 'middle').style('font-size', isPhone ? '10px' : '13px').text(dateMode ? 'Release Date' : 'Average Cost per Run (USD)');
    g.append('text').attr('class', 'axis-label').attr('transform', 'rotate(-90)').attr('x', -innerHeight / 2).attr('y', isPhone ? -33 : -50)
        .attr('text-anchor', 'middle').style('font-size', isPhone ? '10px' : '13px').text('Average Score Across 11 Tasks');
    if (dateMode && data.date_frontiers.undated_models.length) {
        g.append('text').attr('class', 'chart-note').attr('x', 8).attr('y', 15).attr('font-size', 11)
            .text(`${data.date_frontiers.undated_models.length} model(s) without release dates omitted`);
    }
    if (currentMode === 'frontier') {
        const dateX = point => xScale(new Date(point[0] + 'T00:00:00Z'));
        const prepared = data.date_frontiers;
        // Step points and gap values are prepared offline, including the export's as-of date.
        if (prepared.open.length > 1) g.append('path').datum(prepared.open).attr('class', 'frontier-fill')
            .attr('d', d3.area().x(dateX).y0(yScale(0)).y1(p => yScale(p[1])))
            .attr('fill', openColor).attr('opacity', .1);
        if (prepared.gap.length > 1) g.append('path').datum(prepared.gap).attr('class', 'frontier-gap')
            .attr('d', d3.area().x(dateX).y0(p => yScale(p[1])).y1(p => yScale(p[2])))
            .attr('fill', closedColor).attr('opacity', .1);
        for (const [group, stroke] of [['open', openColor], ['closed', closedColor]]) {
            if (prepared[group].length) g.append('path').datum(prepared[group]).attr('class', 'frontier-line')
                .attr('d', d3.line().x(dateX).y(p => yScale(p[1])))
                .attr('fill', 'none').attr('stroke', stroke).attr('stroke-width', 3).attr('opacity', .8);
        }
    }
    // Measure every label, then place them all before drawing, so each sits next to its own marker.
    const probe = g.append('text').attr('class', 'model-label').style('visibility', 'hidden');
    const items = models.map((model, i) => {
        const size = probe.text(model.name).node().getBBox();
        return {model, priority: i, x: xScale(getX(model)), y: yScale(model.score), r: markerRadius(model, isPhone), w: size.width + 4, h: size.height};
    });
    probe.remove();
    // Labels may use the right margin, but stay clear of the axes and their titles.
    placeScatterLabels(items, {x0: 2, x1: innerWidth + margin.right - 4, y0: -margin.top + 2, y1: innerHeight - 2}, isPhone ? 2 : 3);
    items.forEach(({model, x: cx, y: cy, r, box, leader}) => {
        const modelStroke = currentMode === 'frontier' ? (model.open_weights ? openColor : closedColor) : color(model);
        const group = g.append('g').datum({model}).attr('class', 'comparison-model').attr('tabindex', 0).attr('role', 'button')
            .attr('aria-label', `${model.name}, score ${fmtPct(model.score)}`);
        if (leader) {
            // Run from the marker's edge to the nearest point of the label box.
            const tx = Math.min(Math.max(cx, box.x0), box.x1), ty = Math.min(Math.max(cy, box.y0), box.y1);
            const length = Math.hypot(tx - cx, ty - cy) || 1;
            group.append('line').attr('class', 'label-connector').attr('stroke', modelStroke)
                .attr('x1', cx + (tx - cx) / length * (r + 1)).attr('y1', cy + (ty - cy) / length * (r + 1)).attr('x2', tx).attr('y2', ty);
        }
        drawMarker(group, cx, cy, model, modelStroke, isPhone);
        group.append('text').attr('class', 'model-label').attr('x', box.x0 + 2).attr('y', (box.y0 + box.y1) / 2)
            .attr('dominant-baseline', 'central').attr('text-anchor', 'start').attr('fill', modelStroke).text(model.name);
        group.on('pointerenter', event => {if (!pinned) {setHighlight(model.id); modelCard(model); showTooltip(event);}})
            .on('pointermove', event => {if (!pinned) placeTooltip(event);})
            .on('pointerleave', () => {if (!pinned) {setHighlight(null); hideTooltip();}})
            .on('click', event => {
                event.stopPropagation();
                if (currentMode !== 'frontier') {selectFromClick(model, event, null); return;}
                pinned = !pinned; if (pinned) {setHighlight(model.id); modelCard(model); showTooltip(event);} else {setHighlight(null); hideTooltip();}
            })
            .on('keydown', event => {if (event.key === 'Enter' || event.key === ' ') {event.preventDefault(); if (currentMode !== 'frontier') toggleSelected(model.id); setHighlight(model.id); modelCard(model); showTooltip({pageX: cx + margin.left, pageY: cy + margin.top});} if (event.key === 'Escape') {pinned = false; setHighlight(null); hideTooltip();}});
    });
    svg.on('click', () => {pinned = false; setHighlight(null); hideTooltip();});
    if (pngExport) drawSvgLegend(g, innerWidth + margin.right - 10, innerHeight + 68);
}

// ── Task grid: one small token-vs-score panel per model (row) and configuration (column). ──
function updateGrid() {
    const isPhone = isPhoneNow();
    const container = document.getElementById('chart').parentElement;
    const ordered = data.legend_columns.flat().map(id => data.configurations.find(c => c.id === id)).filter(Boolean);
    // Columns: every configuration on its own, or one column per base task with its hint twins overlaid.
    let columns;
    if (currentGrid === 'tasks') {
        const byTask = new Map();
        ordered.forEach(c => {if (!byTask.has(c.task)) byTask.set(c.task, []); byTask.get(c.task).push(c);});
        columns = [...byTask.entries()].map(([task, configs]) => ({
            id: task, task, name: configs[0].name,
            configs: [...configs].sort((a, b) => (b.hint_mode === 'Hints allowed') - (a.hint_mode === 'Hints allowed'))
        }));
    } else {
        columns = ordered.map(c => ({id: c.id, task: c.task, name: c.name, configs: [c]}));
    }
    const models = data.models;
    const fullWidth = container.clientWidth - 20;
    // Row headers live in their own SVG so they can stay pinned while narrow screens pan the cells.
    let heads = d3.select(container).select('svg.grid-heads');
    if (heads.empty()) heads = d3.select(container).insert('svg', '#chart').attr('class', 'grid-heads').attr('aria-hidden', 'true');
    heads.selectAll('*').remove();
    const iconSpace = isPhone ? 22 : 26;
    const labelWidth = isPhone ? 124 : 186;
    const nameWidth = labelWidth - iconSpace - 8;
    const probe = heads.append('text').attr('class', 'grid-model-name').style('visibility', 'hidden');
    const textWidth = t => {probe.text(t); return probe.node().getComputedTextLength();};
    const wrapName = name => {
        if (textWidth(name) <= nameWidth) return [name];
        const words = name.split(' '), lines = [];
        let current = '';
        words.forEach(word => {
            const next = current ? current + ' ' + word : word;
            if (current && textWidth(next) > nameWidth) {lines.push(current); current = word;} else current = next;
        });
        if (current) lines.push(current);
        return lines;
    };
    const nameLines = new Map(models.map(m => [m.id, wrapName(m.name)]));
    probe.remove();
    const gapX = currentGrid === 'tasks' ? 8 : 5, gapY = 7, leftPad = 8;
    const headerHeight = 60, footerHeight = 30;
    const minCell = 54;
    const availableWidth = fullWidth - labelWidth - leftPad - 4;
    const cellWidth = Math.max(minCell, Math.floor((availableWidth - gapX * (columns.length - 1)) / columns.length));
    const cellHeight = Math.min(60, Math.max(48, Math.round(cellWidth * 0.86)));
    const gridWidth = leftPad + columns.length * cellWidth + gapX * (columns.length - 1) + 4;
    const height = headerHeight + models.length * (cellHeight + gapY) - gapY + footerHeight;
    // Inline sizes beat the stylesheet's 100% width so the cells can be wider than the frame.
    heads.attr('width', labelWidth).attr('height', height).style('width', labelWidth + 'px').style('height', height + 'px');
    svg.attr('width', gridWidth).attr('height', height).style('width', gridWidth + 'px').style('height', height + 'px')
        .on('pointermove', null).on('pointerleave', null).on('click', null);
    svg.selectAll('*').remove();
    container.classList.toggle('can-scroll', labelWidth + 10 + gridWidth > container.clientWidth - 10);
    tooltip.classed('mobile-panel', isPhone).classed('visible', false);
    pinned = false; highlighted = null;
    const g = svg.append('g').attr('class', 'task-grid');
    const colX = i => leftPad + i * (cellWidth + gapX);
    const rowY = i => headerHeight + i * (cellHeight + gapY);
    const yScale = d3.scaleLinear().domain([0, 1]).range([cellHeight, 0]);
    const bisect = d3.bisector(p => p[0]).right;
    const twins = new Map();
    columns.forEach((c, i) => {if (!twins.has(c.task)) twins.set(c.task, []); twins.get(c.task).push(i);});

    // Column headers: the task name spans its hinted/hintless pair; the hint mode sits beneath.
    const headers = g.append('g').attr('class', 'grid-headers');
    const nameBaseline = headerHeight - (currentGrid === 'tasks' ? 12 : 22);
    twins.forEach((indices, task) => {
        const first = columns[indices[0]];
        const x0 = colX(indices[0]), x1 = colX(indices.at(-1)) + cellWidth;
        const cx = (x0 + x1) / 2;
        const label = headers.append('text').attr('class', 'grid-task-name').attr('x', cx).attr('text-anchor', 'middle');
        const words = first.name.split(' ');
        const probe2 = headers.append('text').attr('class', 'grid-task-name').style('visibility', 'hidden').text(first.name);
        const fits = probe2.node().getComputedTextLength() <= x1 - x0 - 2;
        probe2.remove();
        if (fits || words.length === 1) {
            label.attr('y', nameBaseline).text(first.name);
        } else {
            const mid = Math.ceil(words.length / 2);
            label.attr('y', nameBaseline - 13);
            label.append('tspan').attr('x', cx).text(words.slice(0, mid).join(' '));
            label.append('tspan').attr('x', cx).attr('dy', 13).text(words.slice(mid).join(' '));
        }
        if (indices.length > 1) {
            headers.append('line').attr('class', 'grid-twin-bracket').attr('x1', x0 + 1).attr('x2', x1 - 1)
                .attr('y1', nameBaseline + 5).attr('y2', nameBaseline + 5);
            indices.forEach(i => {
                headers.append('text').attr('class', 'grid-hint-mode').attr('x', colX(i) + cellWidth / 2).attr('y', headerHeight - 7)
                    .attr('text-anchor', 'middle').text(columns[i].configs[0].hint_mode === 'Hints allowed' ? 'hints' : 'no hints');
            });
        }
    });

    // Row headers: lab icon, model name (wrapped when long) and official score.
    const headRows = heads.selectAll('.grid-row-head').data(models).join('g').attr('class', 'grid-row-head')
        .datum(m => ({model: m})).attr('transform', (d, i) => `translate(0,${rowY(i)})`);
    headRows.each(function ({model}) {
        const head = d3.select(this);
        const icon = icons[company(model)];
        const iconSize = isPhone ? 16 : 18;
        const lines = nameLines.get(model.id);
        const lineHeight = 14, scoreGap = 15;
        const block = lines.length * lineHeight + scoreGap;
        const top = cellHeight / 2 - block / 2 + lineHeight / 2;
        if (icon) head.append('image').attr('href', 'assets/icons/' + icon).attr('x', 0).attr('y', cellHeight / 2 - iconSize / 2)
            .attr('width', iconSize).attr('height', iconSize);
        else head.append('circle').attr('cx', iconSize / 2).attr('cy', cellHeight / 2).attr('r', iconSize / 2 - 3).attr('fill', color(model));
        const name = head.append('text').attr('class', 'grid-model-name').attr('x', iconSpace).attr('y', top).attr('dominant-baseline', 'middle');
        lines.forEach((line, i) => name.append('tspan').attr('x', iconSpace).attr('dy', i ? lineHeight : 0).text(line));
        head.append('text').attr('class', 'grid-model-score').attr('x', iconSpace).attr('y', top + (lines.length - 1) * lineHeight + scoreGap)
            .attr('dominant-baseline', 'middle').text(fmtPct(model.score) + ' overall');
    });

    // Cells. A task column draws its hinted twin solid with a fill and its hintless twin dashed.
    const rows = g.selectAll('.grid-row').data(models).join('g').attr('class', 'grid-row')
        .attr('transform', (m, i) => `translate(0,${rowY(i)})`);
    rows.each(function (model, rowIndex) {
        const row = d3.select(this);
        columns.forEach((column, colIndex) => {
            const entries = column.configs.map(config => ({config, c: model.configurations[config.id]})).filter(e => e.c);
            if (!entries.length) return;
            const axis = entries[0].c.axis;
            const plotStart = currentScale === 'log' ? 100000 * axis.limit / data.overall_axis.limit : 0;
            const xScale = currentScale === 'log'
                ? d3.scaleLog().domain([plotStart, axis.limit]).range([0, cellWidth])
                : d3.scaleLinear().domain([0, axis.limit]).range([0, cellWidth]);
            const cell = row.append('g').attr('class', 'grid-cell').datum({model, config: column})
                .attr('transform', `translate(${colX(colIndex)},0)`);
            cell.append('rect').attr('class', 'grid-cell-bg').attr('width', cellWidth).attr('height', cellHeight).attr('rx', 3);
            if (currentScale === 'log' && axis.start > plotStart) {
                cell.append('rect').attr('class', 'grid-cell-window').attr('x', xScale(axis.start)).attr('y', 0)
                    .attr('width', cellWidth - xScale(axis.start)).attr('height', cellHeight);
            }
            cell.append('line').attr('class', 'grid-cell-mid').attr('x1', 0).attr('x2', cellWidth).attr('y1', yScale(0.5)).attr('y2', yScale(0.5));
            const area = d3.area().x(p => xScale(Math.max(plotStart, p[0]))).y0(yScale(0)).y1(p => yScale(p[1])).curve(d3.curveStepAfter);
            const line = d3.line().x(p => xScale(Math.max(plotStart, p[0]))).y(p => yScale(p[1])).curve(d3.curveStepAfter);
            // Draw the secondary (hintless) twin first so the primary curve sits on top.
            [...entries].reverse().forEach(({config, c}, reverseIndex) => {
                const primary = reverseIndex === entries.length - 1;
                const secondary = entries.length > 1 && !primary;
                const points = c.curve;
                const first = Math.max(0, bisect(points, plotStart) - 1);
                const visible = points.slice(first);
                if (!secondary) cell.append('path').datum(visible).attr('class', 'grid-area').attr('d', area).attr('fill', color(model));
                cell.append('path').datum(visible).attr('class', 'grid-line progress-line' + (secondary ? ' secondary' : '')).attr('d', line).attr('fill', 'none')
                    .attr('stroke', color(model)).attr('stroke-width', secondary ? 1.3 : 1.6).attr('stroke-dasharray', secondary ? '3 2.5' : null);
                const [lastX, lastY] = points.at(-1);
                const end = cell.append('circle').attr('class', 'grid-end').attr('cx', xScale(lastX)).attr('cy', yScale(lastY)).attr('r', 2.4)
                    .attr('stroke', secondary ? color(model) : '#fff').attr('stroke-width', 1);
                end.attr('fill', secondary ? '#fff' : color(model));
            });
            // Curves rise from the bottom-left, so the top-left corner stays clear for the scores.
            const score = cell.append('text').attr('class', 'grid-cell-score').attr('x', 4).attr('y', 4).attr('dominant-baseline', 'hanging');
            entries.forEach(({c}, i) => {
                score.append('tspan').attr('class', i ? 'secondary' : null).text((i ? ' / ' : '') + d3.format('.0%')(c.score));
            });
            cell.append('rect').attr('class', 'grid-cell-hit').attr('width', cellWidth).attr('height', cellHeight).attr('fill', 'transparent');
            if (pngExport) return;
            const describe = entries.map(({config, c}) => `${config.name}${config.hint_mode === 'Hints allowed' ? ' (hints)' : ''} ${fmtPct(c.score)}`).join(', ');
            cell.attr('tabindex', 0).attr('role', 'button').attr('aria-label', `${model.name}: ${describe}`)
                .on('pointerenter', event => {if (!pinned && event.pointerType !== 'touch') {focusCell(model, column); showTooltip(event);}})
                .on('pointermove', event => {if (!pinned) placeTooltip(event);})
                .on('pointerleave', () => {if (!pinned) {focusCell(null); hideTooltip();}})
                .on('click', event => {
                    event.stopPropagation();
                    if (pinned && highlighted === model.id + '/' + column.id) {pinned = false; focusCell(null); hideTooltip(); return;}
                    pinned = true; focusCell(model, column); showTooltip(event);
                })
                .on('keydown', event => {
                    if (event.key === 'Enter' || event.key === ' ') {event.preventDefault(); pinned = true; focusCell(model, column); showTooltip({pageX: labelWidth + colX(colIndex) + cellWidth, pageY: rowY(rowIndex) + cellHeight});}
                    if (event.key === 'Escape') {pinned = false; focusCell(null); hideTooltip();}
                });
        });
    });

    // Footer: a tiny axis under the first column, and a reading note beside it.
    const footerY = rowY(models.length) - gapY + 5;
    const footer = g.append('g').attr('class', 'grid-footer').attr('transform', `translate(0,${footerY})`);
    const axis0 = models[0].configurations[columns[0].configs[0].id].axis;
    const start0 = currentScale === 'log' ? 100000 * axis0.limit / data.overall_axis.limit : 0;
    const mini = footer.append('g').attr('class', 'grid-mini-axis').attr('transform', `translate(${colX(0)},0)`);
    mini.append('line').attr('x1', 0).attr('x2', cellWidth).attr('y1', 0).attr('y2', 0);
    [0, cellWidth].forEach(x => mini.append('line').attr('x1', x).attr('x2', x).attr('y1', 0).attr('y2', 3));
    mini.append('text').attr('class', 'grid-axis-note').attr('x', 0).attr('y', 6).attr('dominant-baseline', 'hanging').text(fmtTokens(start0));
    mini.append('text').attr('class', 'grid-axis-note').attr('x', cellWidth).attr('y', 6).attr('dominant-baseline', 'hanging')
        .attr('text-anchor', 'end').text(fmtTokens(axis0.limit));
    const shipNote = columns.some(c => c.configs.some(k => k.id === 'ship_detect')) ? ' (Ship Detect: cost-weighted, ×25)' : '';
    const twinNote = currentGrid === 'tasks' ? ' · solid: hints, dashed: no hints' : '';
    // The reading note wraps at its separators when the grid is narrower than the sentence.
    const note = footer.append('text').attr('class', 'grid-axis-note').attr('x', colX(1) + 2).attr('y', 6).attr('dominant-baseline', 'hanging');
    const parts = [`x: tokens per task, ${currentScale} scale${shipNote}`, 'y: best-so-far effective score, 0–100%', 'number: task score' + twinNote, 'shading: scoring window'];
    const noteWidth = gridWidth - colX(1) - 6;
    let lineText = '', lineCount = 0;
    const flush = () => {note.append('tspan').attr('x', colX(1) + 2).attr('dy', lineCount ? 13 : 0).text(lineText); lineCount++;};
    parts.forEach(part => {
        const candidate = lineText ? lineText + ' · ' + part : part;
        note.text(candidate);
        if (lineText && note.node().getComputedTextLength() > noteWidth) {note.text(''); flush(); lineText = part;}
        else {note.text(''); lineText = candidate;}
    });
    flush();
    if (lineCount > 1) svg.attr('height', height + (lineCount - 1) * 13).style('height', height + (lineCount - 1) * 13 + 'px');
    if (pngExport) {
        footer.append('text').attr('class', 'svg-attribution').attr('x', gridWidth - 2).attr('y', 6).attr('dominant-baseline', 'hanging')
            .attr('text-anchor', 'end').text('htihle.github.io/weirdml');
    }

    function focusCell(model, column) {
        highlighted = model ? model.id + '/' + column.id : null;
        svg.selectAll('.grid-cell').classed('focus', d => Boolean(model) && d.model.id === model.id && d.config.id === column.id)
            .classed('same-row', d => Boolean(model) && d.model.id === model.id && d.config.id !== column.id)
            .classed('same-col', d => Boolean(model) && d.config.id === column.id && d.model.id !== model.id)
            .classed('dimmed', d => Boolean(model) && d.model.id !== model.id && d.config.id !== column.id);
        heads.selectAll('.grid-row-head').classed('focus', d => Boolean(model) && d.model.id === model.id)
            .classed('dimmed', d => Boolean(model) && d.model.id !== model.id);
        if (model) modelCard(model, column.configs);
    }
    svg.on('click', () => {pinned = false; focusCell(null); hideTooltip();});
}

function updateChart() {
    if (!data || !data.models.length) return;
    svg.style('cursor', null);
    renderChart();
    // Redraws reset hover state; reapply legend selections to the new marks.
    if (currentMode !== 'grid') setHighlight(null);
}
function renderChart() {
    document.body.classList.toggle('grid-mode', currentMode === 'grid');
    updateLegend();
    if (currentMode === 'grid') {updateGrid(); return;}
    d3.select(svg.node().parentElement).select('svg.grid-heads').remove();
    svg.node().parentElement.classList.remove('can-scroll');
    svg.style('width', null).style('height', null);
    if (['cost', 'date', 'frontier'].includes(currentMode)) {updateComparison(); return;}
    const isPhone = isPhoneNow();
    const rightLabels = !isPhone;
    const chipR = pngExport ? 13 : 12; // half-width reserved for the end marker
    // End labels sit right of the plot. Names much longer than usual wrap before their "(effort, provider)" part, so
    // one long name does not squeeze the plot. The margin fits the widest label with some room to grow on hover;
    // a label that would outgrow it on hover is enlarged less (--focus-size).
    const lineHeight = 1.15, focusSize = 14;
    const labelLines = new Map();
    let labelWidth = 0, baseSize = 10;
    if (rightLabels) {
        const probe = svg.append('text').attr('class', 'model-label').style('visibility', 'hidden');
        baseSize = parseFloat(getComputedStyle(probe.node()).fontSize) || 10;
        const measure = text => {probe.text(text); return probe.node().getComputedTextLength();};
        data.models.forEach(m => {
            let lines = [m.name];
            const split = m.name.lastIndexOf(' (');
            if (measure(m.name) > baseSize * 18 && split > 0) lines = [m.name.slice(0, split), m.name.slice(split + 1)];
            const width = d3.max(lines, measure);
            labelLines.set(m.id, {lines, width});
            labelWidth = Math.max(labelWidth, width);
        });
        probe.remove();
        if (!pngExport) labelWidth *= 1.25;
    }
    const container = document.getElementById('chart').parentElement;
    const width = container.clientWidth - 20;
    const rightMargin = rightLabels ? Math.ceil(labelWidth) + chipR + 26 : pngExport ? 28 : 40;
    const legendRows = pngExport ? measureSvgLegend(width - 65 - 10) : 0;
    const margin = pngExport ? {top: 16, right: rightMargin, bottom: 74 + (legendRows - 1) * 20, left: 65} : isPhone
        ? {top: 20, right: 20, bottom: 40, left: 45}
        : {top: 24, right: rightMargin, bottom: 60, left: 65};
    tooltip.classed('mobile-panel', isPhone).classed('visible', false);
    pinned = false; highlighted = null;
    const height = Math.max(isPhone ? 180 : 200, container.clientHeight - 20);
    svg.attr('width', width).attr('height', height);
    svg.selectAll('*').remove();
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
    const config = currentConfig();
    const axis = currentMode === 'overall' ? data.overall_axis : data.models[0].configurations[config.id].axis;
    const series = data.models.map(model => ({model, points: currentMode === 'overall' ? model.overall_curve : model.configurations[config.id].curve}));
    // Use the same budget fraction in every log view (100k ordinary / 4k Ship Detect).
    const plotStart = currentScale === 'log'
        ? 100000 * axis.limit / data.overall_axis.limit : 0;
    const xScale = currentScale === 'log'
        ? d3.scaleLog().domain([plotStart, axis.limit]).range([0, innerWidth])
        : d3.scaleLinear().domain([0, axis.limit * 1.05]).range([0, innerWidth]).nice();
    const yScale = d3.scaleLinear()
        .domain(currentMode === 'task' ? [0, 1]
            : [0, (d3.max(series, s => s.points.at(-1)[1]) || 1) * 1.05])
        .range([innerHeight, 0]).nice();
    const tickCount = isPhone ? 4 : 6;
    const axisFontSize = isPhone ? '9px' : pngExport ? '13px' : '12px';
    const logTicks = () => {
        const steps = isPhone ? [1] : [1, 2, 5], values = [];
        for (let e = Math.floor(Math.log10(plotStart)); e <= Math.ceil(Math.log10(axis.limit)); e++)
            steps.forEach(k => {const v = k * 10 ** e; if (v >= plotStart * .999 && v <= axis.limit * 1.001) values.push(v);});
        if (!values.some(v => v >= axis.limit / 1.6)) values.push(axis.limit);
        return values;
    };
    const makeAxis = (constructor) => currentScale === 'log'
        ? constructor(xScale).tickValues(logTicks()).tickFormat(d3.format('~s'))
        : constructor(xScale).ticks(tickCount, '~s');
    // Scoring window: the part of the token axis that contributes to the official score.
    const hasWindow = currentScale === 'log' && Number.isFinite(axis.start) && axis.start > plotStart;
    if (hasWindow) {
        const x0 = xScale(axis.start), x1 = xScale(axis.limit);
        const band = g.append('g').attr('class', 'scoring-window');
        band.append('rect').attr('x', x0).attr('y', 0).attr('width', Math.max(0, x1 - x0)).attr('height', innerHeight);
        band.append('line').attr('x1', x0).attr('x2', x0).attr('y1', 0).attr('y2', innerHeight);
        band.append('text').attr('x', x0 + 8).attr('y', isPhone ? 11 : 14).text('Scoring window');
    }
    g.append('g').attr('class', 'grid').call(d3.axisLeft(yScale).tickSize(-innerWidth).tickFormat(''));
    g.append('g').attr('class', 'grid').call(makeAxis(d3.axisBottom).tickSize(innerHeight).tickFormat(''));
    g.append('g').attr('class', 'axis axis-x').attr('transform', `translate(0,${innerHeight})`)
        .call(makeAxis(d3.axisBottom)).selectAll('text').style('font-size', axisFontSize);
    g.append('g').attr('class', 'axis axis-y').call(d3.axisLeft(yScale).tickFormat(d3.format('.0%')).ticks(isPhone ? 5 : null))
        .selectAll('text').style('font-size', axisFontSize);
    const xLabel = currentMode === 'overall' ? 'Tokens per Task'
        : axis.kind === 'tokens' ? 'Tokens' : 'Cost-Weighted Tokens';
    g.append('text').attr('class', 'axis-label').attr('x', innerWidth / 2)
        .attr('y', innerHeight + (isPhone ? 30 : 45)).attr('text-anchor', 'middle')
        .style('font-size', isPhone ? '10px' : '13px').text(xLabel);
    g.append('text').attr('class', 'axis-label').attr('transform', 'rotate(-90)')
        .attr('x', -innerHeight / 2).attr('y', isPhone ? -33 : -50).attr('text-anchor', 'middle')
        .style('font-size', isPhone ? '10px' : '13px')
        .text(currentMode === 'overall' ? 'Average Effective Score Across 11 Tasks' : 'Mean Best-So-Far Effective Score');
    const line = d3.line().x(p => xScale(Math.max(plotStart, p[0]))).y(p => yScale(p[1])).curve(d3.curveStepAfter);
    const bisect = d3.bisector(p => p[0]).right;
    const labels = series.map(s => ({model: s.model, y: yScale(s.points.at(-1)[1]) - (rightLabels ? 0 : 11),
        extra: rightLabels ? (labelLines.get(s.model.id).lines.length - 1) * lineHeight * baseSize / 2 : 0}));
    // Phones draw labels inside the plot: keep them clear of the scoring-window caption at the top.
    spreadLabels(labels, rightLabels ? 17 : 11, rightLabels ? 7 : hasWindow ? 24 : 10, innerHeight - (rightLabels ? 7 : 4));
    const labelY = new Map(labels.map(l => [l.model.id, l.y]));
    const seriesGroups = g.selectAll('.series').data(series).join('g').attr('class', 'series');
    seriesGroups.each(function ({model, points}) {
        const sg = d3.select(this);
        // Carry the last score before the visible window to its left edge.
        const first = Math.max(0, bisect(points, plotStart) - 1);
        sg.append('path').datum(points.slice(first)).attr('class', 'progress-line')
            .attr('d', line).attr('fill', 'none').attr('stroke', color(model)).attr('stroke-width', pngExport ? 3.2 : 3)
            .attr('stroke-dasharray', curveDash(model));
        const [lastX, lastY] = points.at(-1);
        const ex = xScale(lastX), ey = yScale(lastY), ly = labelY.get(model.id);
        if (rightLabels && Math.abs(ly - ey) > 4) {
            sg.append('line').attr('class', 'label-connector').attr('x1', ex + chipR + 2).attr('y1', ey)
                .attr('x2', ex + chipR + 8).attr('y2', ly).attr('stroke', color(model));
        }
        drawMarker(sg, ex, ey, model, color(model), isPhone);
        if (rightLabels) {
            const {lines, width} = labelLines.get(model.id);
            const text = sg.append('text').attr('class', 'model-label').attr('x', ex + chipR + 10).attr('y', ly)
                .attr('dominant-baseline', 'middle').attr('text-anchor', 'start')
                .style('--focus-size', Math.min(focusSize, baseSize * labelWidth / width).toFixed(1) + 'px');
            if (lines.length === 1) text.text(model.name);
            else lines.forEach((part, i) => text.append('tspan').attr('x', ex + chipR + 10)
                .attr('dy', `${i ? lineHeight : -(lines.length - 1) * lineHeight / 2}em`).text(part));
        } else {
            sg.append('text').attr('class', 'model-label').attr('x', ex - 11)
                .attr('y', ly).attr('text-anchor', 'end').text(model.name);
        }
    });
    const guide = g.append('g').attr('class', 'guide').style('display', 'none');
    guide.append('line').attr('y1', 0).attr('y2', innerHeight);
    const guideDots = guide.selectAll('circle').data(series).join('circle').attr('r', 4).attr('fill', s => color(s.model)).attr('stroke', '#fff').attr('stroke-width', 1.5);

    function valueAt(points, resource) {
        return points[Math.max(0, bisect(points, resource) - 1)][1];
    }
    function readout(event, resource) {
        tooltip.selectAll('*').remove();
        tooltip.append('div').attr('class', 'tooltip-title')
            .text(`${currentMode === 'overall' ? 'All tasks' : config.name} · ${fmtTokens(resource)} tokens`);
        // Lookup of prepared steps only; no client-side aggregation.
        const ranked = series.map(s => ({s, value: valueAt(s.points, resource)})).sort((a, b) => b.value - a.value);
        ranked.forEach(({s, value}) => {
            const row = tooltip.append('div').attr('class', 'tooltip-row');
            const label = row.append('span').attr('class', 'tooltip-label');
            label.append('span').attr('class', 'tooltip-swatch').style('background', color(s.model));
            label.append('span').text(s.model.name);
            row.append('span').attr('class', 'tooltip-value').text(fmtPct(value));
        });
        showTooltip(event);
    }
    // Nearest curve at the pointer's x, its end marker, or its end label; null if none is close.
    function nearest(event) {
        const [mouseX, mouseY] = d3.pointer(event, g.node());
        const resource = Math.max(plotStart, Math.min(axis.limit, xScale.invert(Math.max(0, Math.min(innerWidth, mouseX)))));
        const onCurve = mouseX >= 0 && mouseX <= innerWidth;
        let best = null, bestDistance = Infinity;
        series.forEach(s => {
            const [ex, ey] = s.points.at(-1);
            const endDistance = Math.hypot(mouseX - xScale(ex), mouseY - yScale(ey));
            const curveDistance = onCurve ? Math.abs(mouseY - yScale(valueAt(s.points, resource))) : Infinity;
            const labelDistance = rightLabels && mouseX > innerWidth + chipR ? Math.abs(mouseY - labelY.get(s.model.id)) : Infinity;
            const distance = Math.min(curveDistance, endDistance, labelDistance);
            if (distance < bestDistance) {bestDistance = distance; best = s;}
        });
        return best && bestDistance <= (isPhone ? 14 : 10) ? best : null;
    }
    function inspect(event) {
        const [mouseX, mouseY] = d3.pointer(event, g.node());
        if (mouseX < -margin.left || mouseX > innerWidth + margin.right || mouseY < -margin.top || mouseY > innerHeight + margin.bottom) {
            leave(); return;
        }
        const resource = Math.max(plotStart, Math.min(axis.limit, xScale.invert(Math.max(0, Math.min(innerWidth, mouseX)))));
        const best = nearest(event);
        svg.style('cursor', best ? 'pointer' : null);
        if (best) {
            guide.style('display', 'none');
            if (highlighted !== best.model.id) {setHighlight(best.model.id); modelCard(best.model, config);}
            showTooltip(event);
            return;
        }
        if (highlighted) setHighlight(null);
        const gx = xScale(resource);
        guide.style('display', null).select('line').attr('x1', gx).attr('x2', gx);
        guideDots.attr('cx', gx).attr('cy', s => yScale(valueAt(s.points, resource)));
        readout(event, resource);
    }
    function leave() {
        svg.style('cursor', null); hideTooltip(); guide.style('display', 'none'); if (highlighted) setHighlight(null);
    }
    svg.on('pointermove', event => {if (!pinned && event.pointerType !== 'touch') inspect(event);})
        .on('pointerleave', () => {if (!pinned) leave();})
        .on('click', event => {
            const hit = nearest(event);
            if (hit) {selectFromClick(hit.model, event, config); if (!pinned) inspect(event); return;}
            pinned = !pinned; if (pinned) inspect(event); else leave();
        });
    if (pngExport) drawSvgLegend(g, innerWidth + margin.right - 10, innerHeight + 68);
}
d3.json('assets/data/weirdml_v3.json').then(prepared => {
    data = prepared;
    document.querySelector('.header h1').textContent = 'WeirdML v3: Interactive Model Comparison' +
        (data.mode === 'synthetic' ? ' — Synthetic Preview' : '');
    if (!data.models.length) {document.querySelector('.header h1').textContent = 'No complete models yet'; return;}
    data.configurations.forEach(config => {
        const option = document.createElement('option'); option.value = config.id;
        option.textContent = config.name + (config.hint_mode === 'Hints allowed' ? ' (hints)' : ''); select.append(option);
    });
    updateChart();
}).catch(error => {document.querySelector('.header h1').textContent = 'Could not load results: ' + error.message;});
document.querySelectorAll('[data-mode]').forEach(button => button.addEventListener('click', () => {
    currentMode = button.dataset.mode;
    document.querySelectorAll('[data-mode]').forEach(b => {b.classList.toggle('active', b === button); b.setAttribute('aria-pressed', String(b === button));});
    document.querySelectorAll('[data-scale]').forEach(button => {
        button.disabled = currentMode === 'date' || currentMode === 'frontier';
        button.style.opacity = button.disabled ? '.5' : '1';
    });
    document.getElementById('task-control').style.display = currentMode === 'task' ? 'flex' : 'none';
    document.getElementById('grid-control').style.display = currentMode === 'grid' ? 'flex' : 'none';
    updateChart();
}));
document.querySelectorAll('[data-grid]').forEach(button => button.addEventListener('click', () => {
    currentGrid = button.dataset.grid;
    document.querySelectorAll('[data-grid]').forEach(b => {b.classList.toggle('active', b === button); b.setAttribute('aria-pressed', String(b === button));});
    updateChart();
}));
document.querySelectorAll('[data-scale]').forEach(button => button.addEventListener('click', () => {
    currentScale = button.dataset.scale;
    document.querySelectorAll('[data-scale]').forEach(b => {b.classList.toggle('active', b === button); b.setAttribute('aria-pressed', String(b === button));});
    updateChart();
}));
select.addEventListener('change', updateChart);
// Escape closes a pinned card first, then clears the legend selection.
document.addEventListener('keydown', event => {
    if (event.key !== 'Escape') return;
    if (!['grid', 'frontier'].includes(currentMode) && !pinned && !tooltip.classed('visible')) selected.clear();
    pinned = false; setHighlight(null); hideTooltip();
});
// Redraw only when the width changes: height changes come from the parent page fitting the
// embed to its content (e.g. the phone card appearing) and must not reset a pinned card.
let resizeTimer, lastWidth = window.innerWidth;
window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {if (window.innerWidth !== lastWidth) {lastWidth = window.innerWidth; updateChart();}}, 150);
});
})();
