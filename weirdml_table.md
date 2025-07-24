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

<!-- PapaParse (CSV parser) -->
<script src="https://cdn.jsdelivr.net/npm/papaparse@5.4.1/papaparse.min.js" integrity="sha384-6y3Kxk6q1cJtKpX3T7..." crossorigin="anonymous"></script>

<style>
.table-scroll{max-width:100%;overflow:auto;border:1px solid #ddd;}
table{border-collapse:collapse;font-size:14px;}
th,td{border:1px solid #ddd;padding:4px 8px;white-space:nowrap;}
thead th{position:sticky;top:0;background:#fff;z-index:3;}
tbody th{position:sticky;left:0;background:#fff;z-index:2;}
/* optional: make corner cell sticky above left column */
thead th:first-child{left:0;z-index:4;}
</style>

<script>
(async () => {
  const url = '{{ "/data/weirdml_data.csv" | relative_url }}?v={{ site.time | date: "%s" }}'; // cache-bust
  const csvText = await fetch(url).then(r => r.text());
  const parsed = Papa.parse(csvText, { header: true, skipEmptyLines: true });
  const rows = parsed.data;
  if (!rows.length) return;

  // Guess the model column
  const keys = Object.keys(rows[0]);
  const modelKey = keys.find(k => k.toLowerCase().includes('model')) || keys[0];

  const models = rows.map(r => r[modelKey]);
  const metrics = keys.filter(k => k !== modelKey);

  const thead = document.querySelector('#csv-table thead');
  const tbody = document.querySelector('#csv-table tbody');

  // Header row (models)
  const headerTr = document.createElement('tr');
  headerTr.appendChild(document.createElement('th')); // empty corner
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
    th.textContent = metric;
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
