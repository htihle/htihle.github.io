---
layout: nothing
title: WeirdML Data Table
permalink: /weirdml_table.html
---

<div class="container">
<div class="header">
  <h4>WeirdML Data</h4>
  <a href="{{ "/data/weirdml_data.csv" | relative_url }}" class="download-btn">📊 Download CSV</a>
</div>
  
  <div class="table-container">
    <table id="csv-table">
      <thead></thead>
      <tbody></tbody>
    </table>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/papaparse@5.4.1/papaparse.min.js" integrity="sha384-6y3Kxk6q1cJtKpX3T7..." crossorigin="anonymous"></script>

<style>
  * {
    box-sizing: border-box;
  }

  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    margin: 0;
    padding: 20px;
    background: #f8f9fa;
    color: #333;
  }

  .container {
    max-width: 100%;
    margin: 0 auto;
  }

  .header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 16px;
  }

  h4 {
    margin: 0;
    color: #2c3e50;
  }

  .download-btn {
    display: inline-block;
    height: fit-content;
    padding: 4px 8px;
    border: 1px solid #5a7da0;
    color: black;
    text-decoration: none;
    border-radius: 6px;
    font-size: 10px;
    font-weight: 500;
    transition: background-color 0.2s;
  }

  .download-btn:hover {
    background: #e8f0f7;
  }

  .table-container {
    background: white;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    overflow: auto;
    max-height: fit-content;
    border: 1px solid #e1e5e9;
  }

  #csv-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    margin-bottom: 0;
    font-size: 11px;
    line-height: 1.3;
  }

  th, td {
    border: 1px solid #e1e5e9;
    padding: 2px 6px;
    text-align: left;
    vertical-align: middle;
    margin: 0;
    line-height: 1.1;
  }

  /* Header styling - model names */
  thead th {
    background: #4a5f7a;
    color: white;
    font-weight: 600;
    position: sticky;
    top: 0;
    z-index: 10;
    font-size: 10px;
    min-width: 80px;
    max-width: 120px;
    word-wrap: break-word;
    white-space: normal;
    line-height: 1.1;
    text-align: center;
    padding: 0px;
  }

  /* Corner cell */
  thead th:first-child {
    background: #3d4f63;
    position: sticky;
    left: 0;
    z-index: 11;
    width: fit-content;
  }

  /* Metric labels - sticky left column */
  tbody th {
    background: #ecf0f1;
    color: #2c3e50;
    font-weight: 600;
    position: sticky;
    left: 0;
    z-index: 9;
    border-right: 2px solid #bdc3c7;
    font-size: 11px;
    width: fit-content;
    white-space: nowrap;
    padding: 0px;
    line-height: 1.1;
  }

  tbody td {
    background: white;
    font-family: 'Monaco', 'Menlo', monospace;
    font-size: 10px;
    text-align: center;
    min-width: 80px;
    max-width: 120px;
    word-wrap: break-word;
    padding: 0px;
    line-height: 1.1;
  }

  /* Enhanced zebra striping with more contrast */
  tbody tr:nth-child(even) td {
    background: #e8e9eb;
  }

  tbody tr:nth-child(even) th {
    background: #d1d5d8;
  }

  .avg-accuracy {
    background-color: rgba(144, 198, 144, 0.3) !important;
    font-weight: 600;
  }

  .std-error {
    background-color: #fff9e6 !important;
    color: #5a8da1;
    font-weight: 600;
  }

  /* Hover effects */
  tbody tr:hover td {
    background: #e8f2fa !important;
  }

  tbody tr:hover th {
    background: #d6e7f5 !important;
  }

  /* Better number formatting */
  .number {
    font-weight: 500;
  }

  .accuracy {
    font-weight: 600;
  }

  .cost {
    color: #4a5cb8;
  }

  .time {
    color: #4a8fa8;
  }
</style>

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
  "API source": "API Source"
};

function getAccuracyColor(percentage) {
  // Convert percentage (0-100) to color gradient
  // 0-50: Muted Red to Muted Yellow
  // 50-100: Muted Yellow to Muted Green
  
  const value = Math.max(0, Math.min(100, percentage));
  
  if (value <= 50) {
    // Muted Red to Muted Yellow
    const ratio = value / 50;
    const r = 180;
    const g = Math.round(120 + (60 * ratio));
    const b = 80;
    return `rgb(${r}, ${g}, ${b})`;
  } else {
    // Muted Yellow to Muted Green
    const ratio = (value - 50) / 50;
    const r = Math.round(180 * (1 - ratio) + 100 * ratio);
    const g = Math.round(180 * (1 - ratio) + 160 * ratio);
    const b = Math.round(80 * (1 - ratio) + 100 * ratio);
    return `rgb(${r}, ${g}, ${b})`;
  }
}

function formatValue(value, metric) {
  if (!value || value === '') return '—';
  
  // Format numbers appropriately
  if (!isNaN(value)) {
    const num = parseFloat(value);
    if (metric.includes('acc') && num <= 1) {
      // Accuracy values - show as percentage with 1 decimal
      return (num * 100).toFixed(1) + '%';
    } else if (metric.includes('cost')) {
      // Cost values - show with appropriate decimal places
      return '$' + num.toFixed(num < 0.01 ? 4 : 2);
    } else if (metric.includes('time')) {
      // Time values
      return num.toFixed(2) + 's';
    } else if (metric.includes('code_len')) {
      // Code length - no decimals
      return Math.round(num).toLocaleString();
    } else {
      // Other numbers
      return num.toFixed(3);
    }
  }
  
  return value;
}

function getCellClass(metric) {
  if (metric === 'avg_acc') return 'number avg-accuracy';
  if (metric === 'avg_acc_standard_error') return 'number std-error';
  if (metric.includes('acc')) return 'number accuracy';
  if (metric.includes('cost')) return 'number cost';
  if (metric.includes('time')) return 'number time';
  return 'number';
}

(async () => {
  try {
    const url = '{{ "/data/weirdml_data.csv" | relative_url }}?v={{ site.time | date: "%s" }}';
    const csvText = await fetch(url).then(r => r.text());
    const parsed = Papa.parse(csvText, { header: true, skipEmptyLines: true });
    const rows = parsed.data;
    
    if (!rows.length) {
      document.querySelector('.table-container').innerHTML = '<p style="padding: 20px; text-align: center; color: #666;">No data available</p>';
      return;
    }

    const keys = Object.keys(rows[0]);
    const modelKey = 'display_name';
    const skipKeys = new Set([modelKey, 'internal_model_name']);
    
    const models = rows.map(r => r[modelKey] || 'Unknown');
    const metrics = keys.filter(k => !skipKeys.has(k));

    const thead = document.querySelector('#csv-table thead');
    const tbody = document.querySelector('#csv-table tbody');

    // Create header row
    const headerTr = document.createElement('tr');
    const cornerTh = document.createElement('th');
    cornerTh.textContent = 'Metric \\ Model';
    headerTr.appendChild(cornerTh);
    
    models.forEach(model => {
      const th = document.createElement('th');
      th.textContent = model;
      headerTr.appendChild(th);
    });
    thead.appendChild(headerTr);

    // Create metric rows
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
        
        // Apply color gradient to accuracy values
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
    document.querySelector('.table-container').innerHTML = '<p style="padding: 20px; text-align: center; color: #e74c3c;">Error loading data</p>';
  }
})();
</script>