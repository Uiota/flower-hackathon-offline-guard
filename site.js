// Inject a floating orb CTA on every page that links to the Command layer
// Skips if a .floating-orb already exists
(function() {
  // --- Theme Manager ---
  const THEME_KEY = 'uiotas-theme';
  const root = document.documentElement;

  function getInitialTheme() {
    try {
      const saved = localStorage.getItem(THEME_KEY);
      if (saved === 'light' || saved === 'dark') return saved;
    } catch (_) { /* ignore */ }
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
      ? 'dark' : 'light';
  }

  function applyTheme(theme) {
    root.setAttribute('data-theme', theme);
    root.classList.add('theme-transition');
    try { localStorage.setItem(THEME_KEY, theme); } catch (_) { /* ignore */ }
  }

  function currentTheme() {
    return root.getAttribute('data-theme') || 'dark';
  }

  function toggleTheme() {
    const next = currentTheme() === 'dark' ? 'light' : 'dark';
    applyTheme(next);
    updateToggleIcon(next);
  }

  // Set theme ASAP to avoid flash (on pages that load this early)
  applyTheme(getInitialTheme());

  // --- Toggle UI Injection ---
  function createIcon(kind) {
    const span = document.createElement('span');
    span.className = 'icon';
    span.setAttribute('aria-hidden', 'true');
    span.innerHTML = kind === 'sun'
      ? '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"></circle><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>'
      : '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/></svg>';
    return span;
  }

  let toggleBtn, iconSpan;
  function updateToggleIcon(theme) {
    if (!iconSpan) return;
    iconSpan.innerHTML = '';
    iconSpan.replaceWith(iconSpan = createIcon(theme === 'dark' ? 'sun' : 'moon'));
  }

  document.addEventListener('DOMContentLoaded', () => {
    // Insert toggle in header
    const header = document.querySelector('header');
    if (header) {
      toggleBtn = document.createElement('button');
      toggleBtn.type = 'button';
      toggleBtn.className = 'theme-toggle';
      toggleBtn.setAttribute('aria-label', `Switch to ${currentTheme() === 'dark' ? 'light' : 'dark'} mode`);
      iconSpan = createIcon(currentTheme() === 'dark' ? 'sun' : 'moon');
      const label = document.createElement('span');
      label.textContent = currentTheme() === 'dark' ? 'Light' : 'Dark';
      toggleBtn.appendChild(iconSpan);
      toggleBtn.appendChild(label);
      toggleBtn.addEventListener('click', () => {
        const next = currentTheme() === 'dark' ? 'light' : 'dark';
        toggleBtn.setAttribute('aria-label', `Switch to ${next === 'dark' ? 'light' : 'dark'} mode`);
        label.textContent = next === 'dark' ? 'Light' : 'Dark';
        toggleTheme();
      });

      // Place after nav if present, else at end of header
      const nav = header.querySelector('nav');
      if (nav) nav.appendChild(toggleBtn); else header.appendChild(toggleBtn);
    }

    // --- Floating orb CTA ---
    if (!document.querySelector('.floating-orb')) {
      const a = document.createElement('a');
      a.href = 'command_layer.html';
      a.className = 'floating-orb';
      a.setAttribute('aria-label', 'Open Command');
      const orb = document.createElement('div');
      orb.className = 'orb';
      a.appendChild(orb);
      document.body.appendChild(a);
    }
  });
})();
