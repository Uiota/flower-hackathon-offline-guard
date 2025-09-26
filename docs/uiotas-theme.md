# UIOTAS Theme System

## Overview
A flexible, token-based theme system with seamless light/dark support. It persists user preference, respects system defaults, and updates CSS variables for minimal rerenders.

## Files
- `components/ThemeToggle.tsx` — Theme context, provider, hooks, toggles, utilities
- `components/ThemeUtils.ts` — Barrel re-exports
- `styles/globals.css` — Core CSS tokens and transitions
- `styles.css` — Static site styles consuming the tokens

## Features
- Persistent theme selection via `localStorage` (`uiotas-theme`)
- System preference bootstrap (`prefers-color-scheme`)
- Live CSS variable updates on `<html>` via `[data-theme]`
- Smooth transitions using `.theme-transition`
- Context API: `useTheme()`, `withTheme()`
- Utilities: `adjustColor()`
- UI: `ThemeToggle`, `AdvancedThemeToggle`, and optional `ThemeDebugger`

## Theme Configuration
`components/ThemeToggle.tsx` exports `themes`:
- `bg`: page background
- `panel`: panels/cards
- `text`: primary text
- `muted`: secondary text
- `orb`: primary visual accent
- `accent`: interactive highlight
- `ok`: success state
- `warn`: warning state
- `gradient?`: optional `{ from, to }`

## Usage
Import from the barrel for ergonomics:

```tsx
import { ThemeProvider, ThemeToggle, useTheme, withTheme } from '@/components/ThemeUtils';
```

Wrap your app (Next.js):

```tsx
// app/layout.tsx
import '@/styles/globals.css';
import { ThemeProvider, ThemeToggle } from '@/components/ThemeUtils';

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <ThemeProvider>
          <nav>
            <div className="logo">UIOTA.SPACE</div>
            <ThemeToggle />
          </nav>
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}
```

Access theme in components:

```tsx
import { useTheme } from '@/components/ThemeUtils';

function MyPanel() {
  const { theme, themeConfig, toggleTheme } = useTheme();
  return (
    <div style={{ background: themeConfig.panel, color: themeConfig.text }}>
      Current theme: {theme}
      <button onClick={toggleTheme}>Switch</button>
    </div>
  );
}
```

Consume CSS variables in styles:

```css
.my-card {
  background: var(--panel);
  color: var(--text);
  border: 1px solid var(--muted);
}
```

## Tokens in CSS
`styles/globals.css` defines dark defaults and light overrides using `[data-theme="light"]`. Additional brand tokens live in `styles.css` for the static site (`--brand-indigo`, `--brand-violet`, `--brand-mint`, `--glass`, `--glass-border`).

## Debugging
Use `ThemeDebugger` during development for quick inspection and switching:

```tsx
import { ThemeDebugger } from '@/components/ThemeDebugger';
{process.env.NODE_ENV === 'development' && <ThemeDebugger />}
```

## Performance
- Variable updates on `<html>` avoid component rerenders
- Preference read/writes are minimal and isolated

## Roadmap
- Add more theme presets
- Theme export/import for design tuning
- Optional per-surface tokens (e.g., `--panel-2`, `--border`, `--ghost`)

