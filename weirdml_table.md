---
layout: nothing
title: WeirdML Data Table
permalink: /weirdml_table.html
---

<link rel="stylesheet" href="{{ '/assets/css/main.css' | relative_url }}">

<style>
  body {
    font-family: var(--font-sans);
    margin: 0;
    padding: var(--space-6);
    background: var(--color-bg);
    color: var(--color-text);
  }

  .page-container {
    max-width: 100%;
    margin: 0 auto;
  }

  .page-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: var(--space-6);
    flex-wrap: wrap;
    gap: var(--space-4);
  }

  .page-header h1 {
    margin: 0;
    font-size: var(--text-2xl);
    color: var(--color-primary);
  }

  .header-links {
    display: flex;
    gap: var(--space-3);
  }

  .back-link {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    padding: var(--space-2) var(--space-4);
    color: var(--color-text-muted);
    font-size: var(--text-sm);
    font-weight: var(--font-medium);
    text-decoration: none;
    border-radius: var(--radius-md);
    transition: all var(--transition-fast);
  }

  .back-link:hover {
    color: var(--color-primary);
    background-color: var(--color-bg-alt);
    text-decoration: none;
  }

  .download-btn {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    padding: var(--space-2) var(--space-4);
    border: 1px solid var(--color-accent);
    color: var(--color-accent);
    text-decoration: none;
    border-radius: var(--radius-md);
    font-size: var(--text-sm);
    font-weight: var(--font-medium);
    transition: all var(--transition-fast);
  }

  .download-btn:hover {
    background: var(--color-accent);
    color: white;
    text-decoration: none;
  }

  .table-wrapper {
    background: var(--color-surface);
    border-radius: var(--radius-xl);
    box-shadow: var(--shadow-md);
    overflow: auto;
    border: 1px solid var(--color-border);
  }

  #csv-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: var(--text-xs);
    line-height: 1.4;
  }

  th, td {
    border: 1px solid var(--color-border);
    padding: var(--space-2) var(--space-3);
    text-align: left;
    vertical-align: middle;
  }

  thead th {
    background: var(--color-primary);
    color: var(--color-text-inverse);
    font-weight: var(--font-semibold);
    position: sticky;
    top: 0;
    z-index: 10;
    font-size: var(--text-xs);
    min-width: 90px;
    max-width: 130px;
    word-wrap: break-word;
    white-space: normal;
    text-align: center;
  }

  thead th:first-child {
    background: var(--color-primary-light);
    position: sticky;
    left: 0;
    z-index: 11;
    min-width: 150px;
  }

  tbody th {
    background: var(--color-bg-alt);
    color: var(--color-primary);
    font-weight: var(--font-semibold);
    position: sticky;
    left: 0;
    z-index: 9;
    border-right: 2px solid var(--color-border);
    white-space: nowrap;
    min-width: 150px;
  }

  tbody td {
    background: var(--color-surface);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    text-align: center;
    min-width: 90px;
    max-width: 130px;
  }

  tbody tr:nth-child(even) td {
    background: var(--color-bg);
  }

  tbody tr:nth-child(even) th {
    background: var(--color-border-light);
  }

  tbody tr:hover td {
    background: var(--color-accent-subtle) !important;
  }

  tbody tr:hover th {
    background: #cffafe !important;
  }

  .avg-accuracy {
    background-color: #d1fae5 !important;
    font-weight: var(--font-bold);
  }

  .std-error {
    background-color: #fef3c7 !important;
    color: var(--color-info);
    font-weight: var(--font-semibold);
  }

  .loading {
    padding: var(--space-12);
    text-align: center;
    color: var(--color-text-muted);
  }

  .error {
    padding: var(--space-12);
    text-align: center;
    color: var(--color-error);
  }
</style>

<div class="page-container">
  <div class="page-header">
    <h1>WeirdML Data</h1>
    <div class="header-links">
      <a href="weirdml.html" class="back-link">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="m15 18-6-6 6-6"/>
        </svg>
        Back to WeirdML
      </a>
      <a href="{{ '/data/weirdml_data.csv' | relative_url }}" class="download-btn">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="7 10 12 15 17 10"/>
          <line x1="12" x2="12" y1="15" y2="3"/>
        </svg>
        Download CSV
      </a>
    </div>
  </div>

  <div class="table-wrapper">
    <table id="csv-table">
      <thead></thead>
      <tbody></tbody>
    </table>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/papaparse@5.4.1/papaparse.min.js"></script>

<script>
const LABELS = {
  shapes_easy_acc: "Shapes (Easy)",
  shapes_hard_acc: "Shapes (Hard)",
  shuffle_easy_acc: "Shuffle (Easy)",
  shuffle_hard_acc: "Shuffle (Hard)",
  digits_unsup_acc: "Digits (Unsupervised)",
  chess_winners_acc: "Chess Winners",
  kolmo_shuffle_acc: "Kolmogorov Shuffle",
  classify_sentences_acc: "Classify Sentences",
  classify_shuffled_acc: "Classify Shuffled",
  insert_patches_acc: "Insert Patches",
  blunders_easy_acc: "Blunders (Easy)",
  blunders_hard_acc: "Blunders (Hard)",
  digits_generalize_acc: "Digits Generalization",
  shapes_variable_acc: "Shapes (Variable)",
  xor_easy_acc: "XOR (Easy)",
  xor_hard_acc: "XOR (Hard)",
  splash_easy_acc: "Splash (Easy)",
  splash_hard_acc: "Splash (Hard)",
  number_patterns_acc: "Number Patterns",
  avg_acc: "Average Accuracy",
  avg_acc_standard_error: "Avg Acc Std. Error",
  cost_per_run_usd: "Cost / Run (USD)",
  code_len_p10: "Code Length P10",
  code_len_p50: "Code Length P50",
  code_len_p90: "Code Length P90",
  exec_time_median_s: "Exec Time Median (s)",
  release_date: "Release Date",
  "API source": "API Source"
};

function getAccuracyColor(percentage) {
  const value = Math.max(0, Math.min(100, percentage));

  if (value <= 30) {
    const ratio = value / 30;
    const r = Math.round(200 - (40 * ratio));
    const g = Math.round(50 + (30 * ratio));
    const b = Math.round(50 + (80 * ratio));
    return `rgb(${r}, ${g}, ${b})`;
  } else if (value <= 50) {
    const ratio = (value - 30) / 20;
    const r = Math.round(160 - (30 * ratio));
    const g = Math.round(80 + (20 * ratio));
    const b = Math.round(130 + (50 * ratio));
    return `rgb(${r}, ${g}, ${b})`;
  } else if (value <= 70) {
    const ratio = (value - 50) / 20;
    const r = Math.round(130 - (50 * ratio));
    const g = Math.round(100 + (40 * ratio));
    const b = Math.round(180 - (20 * ratio));
    return `rgb(${r}, ${g}, ${b})`;
  } else if (value <= 85) {
    const ratio = (value - 70) / 15;
    const r = Math.round(80 - (40 * ratio));
    const g = Math.round(140 + (40 * ratio));
    const b = Math.round(160 - (40 * ratio));
    return `rgb(${r}, ${g}, ${b})`;
  } else {
    const ratio = (value - 85) / 15;
    const r = Math.round(40 - (20 * ratio));
    const g = Math.round(180 + (40 * ratio));
    const b = Math.round(120 - (60 * ratio));
    return `rgb(${r}, ${g}, ${b})`;
  }
}

function formatValue(value, metric) {
  if (!value || value === '') return '—';

  if (!isNaN(value)) {
    const num = parseFloat(value);
    if (metric.includes('acc') && num <= 1) {
      return (num * 100).toFixed(1) + '%';
    } else if (metric.includes('cost')) {
      return '$' + num.toFixed(num < 0.01 ? 4 : 2);
    } else if (metric.includes('time')) {
      return num.toFixed(2) + 's';
    } else if (metric.includes('code_len')) {
      return Math.round(num).toLocaleString();
    } else {
      return num.toFixed(3);
    }
  }

  return value;
}

function getCellClass(metric) {
  if (metric === 'avg_acc') return 'avg-accuracy';
  if (metric === 'avg_acc_standard_error') return 'std-error';
  return '';
}

(async () => {
  const tbody = document.querySelector('#csv-table tbody');
  tbody.innerHTML = '<tr><td class="loading" colspan="100">Loading data...</td></tr>';

  try {
    const url = '{{ "/data/weirdml_data.csv" | relative_url }}?v={{ site.time | date: "%s" }}';
    const csvText = await fetch(url).then(r => r.text());
    const parsed = Papa.parse(csvText, { header: true, skipEmptyLines: true });
    const rows = parsed.data;

    if (!rows.length) {
      document.querySelector('.table-wrapper').innerHTML = '<p class="loading">No data available</p>';
      return;
    }

    const keys = Object.keys(rows[0]);
    const modelKey = 'display_name';
    const skipKeys = new Set([modelKey, 'internal_model_name', 'model_slug']);

    const models = rows.map(r => r[modelKey] || 'Unknown');
    const metrics = keys.filter(k => !skipKeys.has(k));

    const thead = document.querySelector('#csv-table thead');
    tbody.innerHTML = '';

    const headerTr = document.createElement('tr');
    const cornerTh = document.createElement('th');
    cornerTh.textContent = 'Metric';
    headerTr.appendChild(cornerTh);

    models.forEach(model => {
      const th = document.createElement('th');
      th.textContent = model;
      headerTr.appendChild(th);
    });
    thead.appendChild(headerTr);

    metrics.forEach(metric => {
      const tr = document.createElement('tr');
      const th = document.createElement('th');
      th.textContent = LABELS[metric] || metric;
      tr.appendChild(th);

      rows.forEach(row => {
        const td = document.createElement('td');
        const rawValue = row[metric];
        td.textContent = formatValue(rawValue, metric);
        td.className = getCellClass(metric);

        if (metric.includes('acc') && !isNaN(rawValue) && rawValue !== '') {
          const percentage = parseFloat(rawValue) * 100;
          td.style.color = getAccuracyColor(percentage);
        }

        tr.appendChild(td);
      });

      tbody.appendChild(tr);
    });

  } catch (error) {
    console.error('Error loading data:', error);
    document.querySelector('.table-wrapper').innerHTML = '<p class="error">Error loading data</p>';
  }
})();
</script>
