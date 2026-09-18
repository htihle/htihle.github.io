(() => {
  const summary = document.getElementById('v3-summary-iframe');
  const plot = document.querySelector('iframe[title="WeirdML v3 interactive progress plot"]');
  window.addEventListener('message', event => {
    if (event.origin !== location.origin || !Number.isFinite(event.data?.height)) return;
    const frame = event.data.type === 'v3SummaryHeight' ? summary
      : event.data.type === 'v3PlotHeight' ? plot : null;
    if (frame && event.source === frame.contentWindow) {
      frame.style.setProperty('height', Math.max(150, event.data.height) + 'px', 'important');
    }
  });
})();
