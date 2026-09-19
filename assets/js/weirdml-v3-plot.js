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
const companies = {openai: 'OpenAI', anthropic: 'Anthropic', google: 'Google', deepseek: 'DeepSeek', 'z-ai': 'ZhipuAI'};
const icons = {OpenAI: 'openai_icon.png', Anthropic: 'claude_logo.png', Google: 'gemini_logo.png', DeepSeek: 'deepseek_whale.png', ZhipuAI: 'z-ai_logo.png'};
const company = model => model.id === 'synthetic-strong' ? 'Synthetic Strong' : model.id === 'synthetic-weak' ? 'Synthetic Weak' : companies[model.slug.split('/')[0]] || 'Other';
const color = model => companyColors[company(model)];
const pngExport = new URLSearchParams(location.search).has('png');
const svg = d3.select('#chart');
const tooltip = d3.select('#tooltip');
let data, currentMode = 'overall', currentScale = 'log', pinned = false, highlighted = null;
const select = document.getElementById('task-select');
const openColor = '#1E90FF';
const closedColor = '#2E8B57';
const fmtPct = d3.format('.1%'), fmtUsd = d3.format('$.2f'), fmtTokens = d3.format('.3~s');
const isPhoneNow = () => window.innerWidth <= 600;

// Rank within each lab by official overall score, keeping styles stable across tasks.
function curveDash(model) {
    const lab = model.slug.split('/')[0];
    const peers = data.models.filter(m => m.slug.split('/')[0] === lab)
        .sort((a, b) => b.score - a.score || a.id.localeCompare(b.id));
    const rank = peers.findIndex(m => m.id === model.id);
    const patterns = [null, '8 4', '2 4', '8 4 2 4', '8 4 2 4 2 4'];
    return rank < patterns.length ? patterns[rank] : `12 4 ${'2 4 '.repeat(rank - 2).trim()}`;
}
const harnessNames = {codex_cli: 'Codex CLI', claude_code: 'Claude Code', gemini_cli: 'Gemini CLI', opencode: 'OpenCode'};
function harnessLabel(model) {
    const list = Array.isArray(model.harnesses) ? model.harnesses : [];
    const names = list.map(h => typeof h === 'string' ? (harnessNames[h] || h)
        : [harnessNames[h.name] || h.name, h.version ? 'v' + String(h.version).replace(/^v/, '') : ''].filter(Boolean).join(' '));
    return names.length ? names.join(', ') : null;
}

// ── Highlighting: one model in focus, the rest recede. ──
function setHighlight(id) {
    highlighted = id;
    svg.selectAll('.series, .comparison-model')
        .classed('dimmed', d => Boolean(id) && d.model.id !== id)
        .classed('focus', d => Boolean(id) && d.model.id === id);
    d3.select('#legend').selectAll('.legend-item')
        .classed('dimmed', d => Boolean(id) && d && d.id !== id)
        .classed('focus', d => Boolean(id) && d && d.id === id);
}

// ── Model card: shown when a curve, marker or legend entry is hovered. ──
function modelCard(model, config) {
    tooltip.selectAll('*').remove();
    const head = tooltip.append('div').attr('class', 'tooltip-title');
    head.append('span').attr('class', 'tooltip-swatch').style('background', color(model));
    head.append('span').text(model.name);
    const rows = [];
    if (config) {
        const c = model.configurations[config.id];
        rows.push([config.name + (config.hint_mode === 'Hints allowed' ? ' (hints)' : ''), null, 'section']);
        rows.push(['Task score', fmtPct(c.score)]);
        rows.push(['Final best', fmtPct(c.final_best)]);
        rows.push(['Runs', String(c.n)]);
        if (Number.isFinite(c.mean_api_cost_usd)) rows.push(['Cost / run', fmtUsd(c.mean_api_cost_usd)]);
        rows.push(['Overall', null, 'section']);
    }
    rows.push(['Official score', fmtPct(model.score), 'strong']);
    rows.push(['95% interval', model.interval.map(fmtPct).join(' – ')]);
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
    if (top + h > window.innerHeight + window.scrollY - 12) top = Math.max(8, event.pageY - h - 16);
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
function drawMarker(parent, cx, cy, model, stroke, isPhone) {
    const icon = icons[company(model)];
    const marker = parent.append('g').attr('class', 'model-marker').attr('transform', `translate(${cx},${cy})`);
    if (icon) {
        const base = pngExport ? 24 : isPhone ? 16 : 22;
        const size = company(model) === 'OpenAI' ? base * 1.6 : base;
        marker.append('image').attr('class', 'model-icon').attr('href', 'assets/icons/' + icon)
            .attr('x', -size / 2).attr('y', -size / 2).attr('width', size).attr('height', size);
    } else {
        marker.append('circle').attr('class', 'model-dot').attr('r', isPhone ? 5 : 8)
            .attr('fill', stroke).attr('stroke', '#fff').attr('stroke-width', 1.5);
    }
    return marker;
}
// Spread end labels apart vertically so nearby finishes stay legible.
function spreadLabels(items, minGap, lower, upper) {
    items.sort((a, b) => a.y - b.y);
    for (let i = 1; i < items.length; i++) if (items[i].y - items[i - 1].y < minGap) items[i].y = items[i - 1].y + minGap;
    for (let i = items.length - 1; i >= 0; i--) {
        const cap = i === items.length - 1 ? upper : items[i + 1].y - minGap;
        if (items[i].y > cap) items[i].y = cap;
    }
    for (let i = 0; i < items.length; i++) {
        const floor = i === 0 ? lower : items[i - 1].y + minGap;
        if (items[i].y < floor) items[i].y = floor;
    }
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
    if (currentMode === 'frontier') {
        [{label: 'Closed Frontier', color: closedColor}, {label: 'Open Frontier', color: openColor},
         {label: 'Closed Model', color: closedColor, dot: true}, {label: 'Open Model', color: openColor, dot: true}].forEach(entry => {
            const item = legend.append('div').attr('class', 'legend-item').datum(null);
            if (entry.dot) item.append('span').attr('class', 'legend-dot').style('background', entry.color);
            else item.append('svg').attr('width', 24).attr('height', 14).append('line')
                .attr('x1', 0).attr('x2', 24).attr('y1', 7).attr('y2', 7).attr('stroke', entry.color).attr('stroke-width', 3);
            item.append('span').text(entry.label);
        });
        return;
    }
    data.models.forEach(model => {
        const item = legend.append('div').attr('class', 'legend-item').datum(model).attr('tabindex', pngExport ? null : 0);
        const icon = icons[company(model)];
        if (icon) item.append('img').attr('class', 'legend-icon').attr('src', 'assets/icons/' + icon).attr('alt', '');
        else item.append('span').attr('class', 'legend-dot').style('background', color(model));
        if (currentMode === 'overall' || currentMode === 'task') {
            item.append('svg').attr('width', 30).attr('height', 14).append('line')
                .attr('x1', 1).attr('x2', 29).attr('y1', 7).attr('y2', 7)
                .attr('stroke', color(model)).attr('stroke-width', 3)
                .attr('stroke-dasharray', curveDash(model));
        }
        item.append('span').text(model.name);
        if (pngExport) return;
        item.on('pointerenter', event => {if (!pinned && event.pointerType !== 'touch') {setHighlight(model.id); modelCard(model, currentConfig()); showTooltip(event);}})
            .on('pointermove', event => {if (!pinned) placeTooltip(event);})
            .on('pointerleave', () => {if (!pinned) {setHighlight(null); hideTooltip();}})
            // A tap on a legend entry pins that model's card (hover is not available on touch).
            .on('click', event => {
                if (!isPhoneNow()) return;
                if (pinned && highlighted === model.id) {dismiss(); return;}
                pinned = true; setHighlight(model.id); modelCard(model, currentConfig()); showTooltip(event);
            })
            .on('focus', () => {if (!pinned) setHighlight(model.id);})
            .on('blur', () => {if (!pinned) setHighlight(null);});
    });
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
    const labels = models.map(model => ({model, y: yScale(model.score) - 16}));
    spreadLabels(labels, isPhone ? 11 : 15, 8, innerHeight - 4);
    const labelY = new Map(labels.map(l => [l.model.id, l.y]));
    models.forEach(model => {
        const cx = xScale(getX(model)), cy = yScale(model.score);
        const modelStroke = currentMode === 'frontier' ? (model.open_weights ? openColor : closedColor) : color(model);
        const group = g.append('g').datum({model}).attr('class', 'comparison-model').attr('tabindex', 0).attr('role', 'button')
            .attr('aria-label', `${model.name}, score ${fmtPct(model.score)}`);
        drawMarker(group, cx, cy, model, modelStroke, isPhone);
        const rightSide = cx > innerWidth * .65;
        group.append('text').attr('class', 'model-label').attr('x', cx + (rightSide ? -15 : 15)).attr('y', labelY.get(model.id))
            .attr('text-anchor', rightSide ? 'end' : 'start').attr('fill', modelStroke).text(model.name);
        group.on('pointerenter', event => {if (!pinned) {setHighlight(model.id); modelCard(model); showTooltip(event);}})
            .on('pointermove', event => {if (!pinned) placeTooltip(event);})
            .on('pointerleave', () => {if (!pinned) {setHighlight(null); hideTooltip();}})
            .on('click', event => {event.stopPropagation(); pinned = !pinned; if (pinned) {setHighlight(model.id); modelCard(model); showTooltip(event);} else {setHighlight(null); hideTooltip();}})
            .on('keydown', event => {if (event.key === 'Enter' || event.key === ' ') {event.preventDefault(); setHighlight(model.id); modelCard(model); showTooltip({pageX: cx + margin.left, pageY: cy + margin.top});} if (event.key === 'Escape') {pinned = false; setHighlight(null); hideTooltip();}});
    });
    svg.on('click', () => {pinned = false; setHighlight(null); hideTooltip();});
    if (pngExport) drawSvgLegend(g, innerWidth + margin.right - 10, innerHeight + 68);
}

function updateChart() {
    if (!data || !data.models.length) return;
    updateLegend();
    if (['cost', 'date', 'frontier'].includes(currentMode)) {updateComparison(); return;}
    const isPhone = isPhoneNow();
    const rightLabels = !isPhone;
    const chipR = pngExport ? 13 : 12; // half-width reserved for the end marker
    let labelWidth = 0;
    if (rightLabels) {
        const probe = svg.append('text').attr('class', 'model-label').style('font-size', '14px').style('visibility', 'hidden');
        data.models.forEach(m => {probe.text(m.name); labelWidth = Math.max(labelWidth, probe.node().getComputedTextLength());});
        probe.remove();
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
    const labels = series.map(s => ({model: s.model, y: yScale(s.points.at(-1)[1]) - (rightLabels ? 0 : 11)}));
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
            sg.append('text').attr('class', 'model-label').attr('x', ex + chipR + 10).attr('y', ly)
                .attr('dominant-baseline', 'middle').attr('text-anchor', 'start').text(model.name);
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
    function inspect(event) {
        const [mouseX, mouseY] = d3.pointer(event, g.node());
        if (mouseX < -margin.left || mouseX > innerWidth + margin.right || mouseY < -margin.top || mouseY > innerHeight + margin.bottom) {
            leave(); return;
        }
        const resource = Math.max(plotStart, Math.min(axis.limit, xScale.invert(Math.max(0, Math.min(innerWidth, mouseX)))));
        // Nearest curve at this x, or the end marker under the pointer, takes focus.
        let best = null, bestDistance = Infinity;
        series.forEach(s => {
            const y = yScale(valueAt(s.points, resource));
            const [ex, ey] = s.points.at(-1);
            const endDistance = Math.hypot(mouseX - xScale(ex), mouseY - yScale(ey));
            const distance = Math.min(Math.abs(mouseY - y), endDistance);
            if (distance < bestDistance) {bestDistance = distance; best = s;}
        });
        if (best && bestDistance <= (isPhone ? 14 : 10)) {
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
        hideTooltip(); guide.style('display', 'none'); if (highlighted) setHighlight(null);
    }
    svg.on('pointermove', event => {if (!pinned && event.pointerType !== 'touch') inspect(event);})
        .on('pointerleave', () => {if (!pinned) leave();})
        .on('click', event => {pinned = !pinned; if (pinned) inspect(event); else leave();});
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
    updateChart();
}));
document.querySelectorAll('[data-scale]').forEach(button => button.addEventListener('click', () => {
    currentScale = button.dataset.scale;
    document.querySelectorAll('[data-scale]').forEach(b => {b.classList.toggle('active', b === button); b.setAttribute('aria-pressed', String(b === button));});
    updateChart();
}));
select.addEventListener('change', updateChart);
document.addEventListener('keydown', event => {if (event.key === 'Escape') {pinned = false; setHighlight(null); hideTooltip();}});
// Redraw only when the width changes: height changes come from the parent page fitting the
// embed to its content (e.g. the phone card appearing) and must not reset a pinned card.
let resizeTimer, lastWidth = window.innerWidth;
window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {if (window.innerWidth !== lastWidth) {lastWidth = window.innerWidth; updateChart();}}, 150);
});
})();
