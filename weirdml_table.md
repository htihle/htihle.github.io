---
layout: nothing
title: WeirdML Data Table
permalink: /weirdml_table.html
---

<style>
  * {
    box-sizing: border-box;
  }

  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
    margin: 0;
    padding: 15px;
    background: #f8f9fa;
    color: #333;
  }

  .header {
    position: relative;
    text-align: center;
    margin-bottom: 15px;
  }

  .header h1 {
    margin: 0;
    font-size: 1.4rem;
    color: #2c3e50;
  }

  .header-links {
    position: absolute;
    top: 0;
    left: 0;
    display: flex;
    gap: 12px;
  }

  .header-links a {
    color: #3498db;
    text-decoration: none;
    font-size: 0.85rem;
  }

  .header-links a:hover {
    text-decoration: underline;
  }

  .table-wrapper {
    background: white;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    overflow: auto;
  }

  #csv-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.8rem;
    line-height: 1.4;
  }

  th, td {
    border: 1px solid #e9ecef;
    padding: 8px 12px;
    text-align: left;
    vertical-align: middle;
  }

  thead th {
    background: #2c3e50;
    color: white;
    font-weight: 600;
    position: sticky;
    top: 0;
    z-index: 10;
    font-size: 0.75rem;
    min-width: 90px;
    max-width: 130px;
    word-wrap: break-word;
    white-space: normal;
    text-align: center;
  }

  thead th:first-child {
    background: #34495e;
    position: sticky;
    left: 0;
    z-index: 11;
    min-width: 150px;
  }

  tbody th {
    background: #f8f9fa;
    color: #2c3e50;
    font-weight: 600;
    position: sticky;
    left: 0;
    z-index: 9;
    border-right: 2px solid #dee2e6;
    white-space: nowrap;
    min-width: 150px;
  }

  tbody td {
    background: white;
    font-family: 'SF Mono', 'Fira Code', 'Fira Mono', Menlo, Monaco, Consolas, monospace;
    font-size: 0.75rem;
    text-align: center;
    min-width: 90px;
    max-width: 130px;
  }

  tbody tr:nth-child(even) td {
    background: #f8f9fa;
  }

  tbody tr:nth-child(even) th {
    background: #ecf0f1;
  }

  tbody tr:hover td {
    background: #ecf6fd !important;
  }

  tbody tr:hover th {
    background: #d6eaf8 !important;
  }

  .avg-accuracy {
    background-color: #d1fae5 !important;
    font-weight: 700;
  }

  .std-error {
    background-color: #fef3c7 !important;
    color: #3b82f6;
    font-weight: 600;
  }

  .loading {
    padding: 48px;
    text-align: center;
    color: #666;
  }

  .error {
    padding: 48px;
    text-align: center;
    color: #e74c3c;
  }

  @media (max-width: 1200px), (max-height: 500px) {
    body {
      padding: 8px;
    }

    .header {
      margin-bottom: 8px;
    }

    .header h1 {
      font-size: 1.1rem;
    }

    .header-links {
      position: static;
      justify-content: center;
      margin-top: 6px;
      gap: 10px;
    }

    .header-links a {
      font-size: 0.75rem;
    }
  }
</style>

<div class="header">
  <h1>WeirdML: Data Table</h1>
  <div class="header-links">
    <a href="weirdml.html">&larr; Back to WeirdML</a>
    <a href="weirdml_interactive.html">Interactive plot</a>
    <a href="data/weirdml_data.csv">CSV</a>
  </div>
</div>

<div class="table-wrapper">
  <table id="csv-table">
    <thead></thead>
    <tbody></tbody>
  </table>
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
