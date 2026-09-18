(() => {
  const dialog = document.getElementById('task-lightbox');
  const image = document.getElementById('lightbox-image');
  const caption = document.getElementById('lightbox-caption');
  const sizeButton = document.getElementById('figure-size');
  let opener;
  document.querySelectorAll('.figure-open').forEach(button => {
    const title = button.closest('.task').querySelector('h2').textContent;
    const figure = button.closest('figure');
    button.setAttribute('aria-label', `Enlarge ${title} figure: ${figure.querySelector('figcaption').textContent.split(':')[0].slice(0, 90)}`);
    button.addEventListener('click', () => {
      opener = button;
      const source = button.querySelector('img');
      image.src = source.src;
      image.alt = source.alt;
      caption.textContent = figure.querySelector('figcaption').textContent;
      dialog.classList.remove('actual-size');
      sizeButton.textContent = 'Actual size';
      sizeButton.setAttribute('aria-pressed', 'false');
      dialog.showModal();
      document.getElementById('figure-close').focus();
    });
  });
  sizeButton.addEventListener('click', () => {
    const expanded = dialog.classList.toggle('actual-size');
    sizeButton.setAttribute('aria-pressed', String(expanded));
    sizeButton.textContent = expanded ? 'Fit to screen' : 'Actual size';
  });
  document.getElementById('figure-close').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => { if (event.target === dialog) {
    const rect = dialog.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
  }});
  dialog.addEventListener('close', () => { image.removeAttribute('src'); opener?.focus(); });
})();
