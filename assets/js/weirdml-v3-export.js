// Local-only PNG rendering. A separate view gives exports a stable desktop layout.
(() => {
  const exporting = new URLSearchParams(location.search).has('png');
  if (exporting) document.documentElement.classList.add('png-export');
  if (!exporting && document.body.classList.contains('v3-plot') && window.parent !== window) {
    new ResizeObserver(() => {
      window.parent.postMessage({type: 'v3PlotHeight', height: Math.ceil(document.body.getBoundingClientRect().height)}, location.origin);
    }).observe(document.body);
  }
  const button = document.createElement('button');
  button.className = 'png-button';
  button.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3v12m-4-4 4 4 4-4M5 16v4h14v-4"/></svg><span>Export PNG</span>';
  button.setAttribute('aria-label', 'Export as PNG');
  button.type = 'button';
  const status = document.createElement('span');
  status.className = 'png-status';
  status.setAttribute('role', 'status');
  const toolbar = document.createElement('div');
  toolbar.className = 'png-toolbar';
  toolbar.append(button, status);
  document.querySelector('.header').append(toolbar);
  let fontCssPromise;
  function embeddedFontCss() {
    if (!fontCssPromise) fontCssPromise = (async () => {
      const link = document.querySelector('link[href*="weirdml-v3-fonts.css"]') || document.querySelector('link[href*="fonts.googleapis.com"]');
      if (!link) return '';
      const sheet = await (await fetch(link.href)).text();
      const faces = [];
      const pattern = /@font-face\s*{([^}]*)}/g;
      let match;
      while ((match = pattern.exec(sheet))) {
        const body = match[1];
        const range = /unicode-range:\s*([^;]+)/.exec(body)?.[1] || '';
        if (range && !/U\+0000-00FF/i.test(range)) continue;
        const url = /url\(([^)]+)\)/.exec(body)?.[1]?.replace(/["']/g, '');
        if (!url) continue;
        const blob = await (await fetch(new URL(url, link.href))).blob();
        const data = await new Promise(resolve => {
          const reader = new FileReader(); reader.onload = () => resolve(reader.result); reader.readAsDataURL(blob);
        });
        faces.push('@font-face{' + body.replace(/src:[^;]+;/, `src:url(${data}) format('woff2');`) + '}');
      }
      return faces.join('\n');
    })().catch(() => '');
    return fontCssPromise;
  }
  button.addEventListener('click', async () => {
    button.disabled = true;
    status.textContent = 'Preparing image…';
    let frame;
    try {
      const plot = document.body.classList.contains('v3-plot');
      const mode = document.querySelector('[data-mode].active')?.dataset.mode;
      const scale = document.querySelector('[data-scale].active')?.dataset.scale;
      const task = document.querySelector('#task-select')?.value;
      const grid = document.querySelector('[data-grid].active')?.dataset.grid;
      frame = document.createElement('iframe');
      frame.style.cssText = `position:fixed;left:-10000px;top:0;width:${plot ? 1200 : 1440}px;height:${plot ? 675 : 1000}px;border:0;`;
      frame.setAttribute('aria-hidden', 'true');
      frame.tabIndex = -1;
      const url = new URL(location.href);
      url.search = '?png';
      const loaded = new Promise((resolve, reject) => {
        frame.onload = resolve;
        frame.onerror = reject;
      });
      frame.src = url.href;
      document.body.append(frame);
      await loaded;
      const win = frame.contentWindow, doc = win.document;
      await new Promise((resolve, reject) => {
        const start = Date.now();
        const check = () => {
          if (doc.querySelector(plot ? '.progress-line' : '#table-body tr svg')) return resolve();
          if (Date.now() - start > 20000) return reject(new Error('Results did not load.'));
          setTimeout(check, 50);
        };
        check();
      });
      if (plot) {
        doc.querySelector(`[data-mode="${mode}"]`).click();
        doc.querySelector(`[data-scale="${scale}"]`).click();
        if (grid) doc.querySelector(`[data-grid="${grid}"]`)?.click();
        doc.querySelector('#task-select').value = task;
        doc.querySelector('#task-select').dispatchEvent(new win.Event('change'));
        const label = mode === 'task' ? doc.querySelector('#task-select').selectedOptions[0].textContent
          : {overall:'Score vs tokens', cost:'Score vs API cost', date:'Score vs release date', frontier:'Open vs closed weights', grid: grid === 'tasks' ? 'Score vs tokens, 11 tasks' : 'Score vs tokens, 15 configurations'}[mode];
        doc.querySelector('.header h1').textContent = `WeirdML v3 · ${label}`;
      }
      await doc.fonts.ready;
      // SVGs are rasterised as standalone images, which cannot reach the page's web fonts:
      // inline the latin faces as data URIs so chart text keeps the site typography.
      try {
        const css = await embeddedFontCss();
        if (css) doc.querySelectorAll('svg').forEach(svg => {
          const style = doc.createElementNS('http://www.w3.org/2000/svg', 'style');
          style.textContent = css;
          svg.prepend(style);
        });
      } catch (error) { console.warn('Font embedding skipped:', error); }
      // Inline SVG image references so browser rasterization keeps the lab icons.
      await Promise.all([...doc.querySelectorAll('svg image')].map(async image => {
        const response = await fetch(new URL(image.getAttribute('href'), location.href));
        if (!response.ok) throw new Error('Could not load a chart icon.');
        const blob = await response.blob();
        const value = await new Promise(resolve => {
          const reader = new FileReader(); reader.onload = () => resolve(reader.result); reader.readAsDataURL(blob);
        });
        image.setAttribute('href', value);
      }));
      await Promise.all([...doc.images].map(img => img.decode().catch(() => {})));
      await new Promise(resolve => win.requestAnimationFrame(() => win.requestAnimationFrame(resolve)));
      doc.querySelectorAll('.tick-overlay svg').forEach(svg => {
        const labels = svg.querySelectorAll('text');
        labels[0]?.setAttribute('text-anchor', 'start');
        labels[labels.length - 1]?.setAttribute('text-anchor', 'end');
      });
      // The task grid grows with its rows, so its export takes the rendered height.
      const height = plot ? (mode === 'grid' ? Math.ceil(doc.body.scrollHeight) : 675)
        : Math.ceil(doc.querySelector('#table-wrap').getBoundingClientRect().bottom + 32);
      if (mode === 'grid') frame.style.height = height + 'px';
      const canvas = await win.html2canvas(doc.body, {
        scale: 2, width: plot ? 1200 : 1440, height,
        backgroundColor: '#ffffff', logging: false,
        windowWidth: plot ? 1200 : 1440, windowHeight: plot ? (mode === 'grid' ? height : 675) : 1000
      });
      const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
      if (!blob) throw new Error('Image could not be generated.');
      const link = document.createElement('a');
      link.download = `weirdml-${plot ? mode + (mode === 'task' ? '-' + task : mode === 'grid' ? '-' + grid : '') : 'summary'}.png`;
      link.href = URL.createObjectURL(blob);
      link.click();
      setTimeout(() => URL.revokeObjectURL(link.href), 30000);
      status.textContent = 'PNG exported';
    } catch (error) {
      status.textContent = 'Export failed. Please try again.';
      console.error('PNG export:', error);
    } finally {
      frame?.remove();
      button.disabled = false;
    }
  });
})();
