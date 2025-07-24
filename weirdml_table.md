---
layout: default
title: WeirdML Data Table
permalink: /weirdml_table.html
---

<h1>WeirdML Data (Interactive Table)</h1>
<p><a href="{{ "/data/weirdml_data.csv" | relative_url }}">Download CSV</a></p>

<div class="table-scroll">
  <table id="csv-table">
    <thead></thead>
    <tbody></tbody>
  </table>
</div>

<script src="https://cdn.jsdelivr.net/npm/papaparse@5.4.1/papaparse.min.js" integrity="sha384-6y3Kxk6q1cJtKpX3T7..." crossorigin="anonymous"></script>

<style>
.table-scroll{max-width:100%;overflow:auto;border:1px solid #ddd;}
table{border-collapse:collapse;font-size:14px;}
th,td{border:1px solid #ddd;padding:4px 8px;white-space:nowrap;}
thead th{position:sticky;top:0;background:#fff;z-index:3;}
tbody th{position:sticky;left:0;background:#fff;z-index:2;}
thead th:first-child{left:0;z-index:4;}
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

(async () => {
  const url = '{{ "/data/weirdml_data.csv" | relative_url }}?v={{ site.time | date: "%s" }}';
  const csvText = await fetch(url).then(r => r.text());
  const parsed = Papa.parse(csvText, { header: true, skipEmptyLines: true });
  const rows = parsed.data;
  if (!rows.length) return;

  // Keys from CSV
  const keys = Object.keys(rows[0]);

  // Use 'display_name' as model column, ignore 'internal_model_name'
  const modelKey = 'display_name';
  const skipKeys = new Set([modelKey, 'internal_model_name']);

  const models = rows.map(r => r[modelKey]);
  const metrics = keys.filter(k => !skipKeys.has(k));

  const thead = document.querySelector('#csv-table thead');
  const tbody = document.querySelector('#csv-table tbody');

  // Header row (models)
  const headerTr = document.createElement('tr');
  headerTr.appendChild(document.createElement('th')); // corner
  models.forEach(m => {
    const th = document.createElement('th');
    th.textContent = m;
    headerTr.appendChild(th);
  });
  thead.appendChild(headerTr);

  // Metric rows
  metrics.forEach(metric => {
    const tr = document.createElement('tr');
    const th = document.createElement('th');
    th.textContent = LABELS[metric] || metric; // fallback to raw if missing
    tr.appendChild(th);

    rows.forEach(r => {
      const td = document.createElement('td');
      td.textContent = r[metric];
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
})();
</script>
