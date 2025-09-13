"use client";
import React, { createContext, useContext, useEffect, useMemo, useState } from "react";

export interface ThemeConfig {
  bg: string;
  panel: string;
  text: string;
  muted: string;
  orb: string;
  accent: string;
  ok: string;
  warn: string;
  gradient?: { from: string; to: string };
}

export const themes: Record<"light" | "dark", ThemeConfig> = {
  dark: {
    bg: "#0b0f14",
    panel: "#121821",
    text: "#e6edf6",
    muted: "#9fb0c3",
    orb: "#47d7ff",
    accent: "#76ffe1",
    ok: "#2ee59d",
    warn: "#ffc861",
    gradient: { from: "#11223360", to: "transparent" },
  },
  light: {
    bg: "#f4f4f5",
    panel: "#ffffff",
    text: "#0b0f14",
    muted: "#6b7280",
    orb: "#0077ff",
    accent: "#00c4cc",
    ok: "#10b981",
    warn: "#f59e0b",
    gradient: { from: "#e0e7ff60", to: "transparent" },
  },
};

type Theme = "light" | "dark";

interface ThemeContextType {
  theme: Theme;
  toggleTheme: () => void;
  setTheme: (t: Theme) => void;
  themeConfig: ThemeConfig;
}

const ThemeContext = createContext<ThemeContextType>({
  theme: "dark",
  toggleTheme: () => {},
  setTheme: () => {},
  themeConfig: themes.dark,
});

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setThemeState] = useState<Theme>("dark");

  // initial from storage or system preference
  useEffect(() => {
    const saved = (typeof window !== "undefined" && localStorage.getItem("uiotas-theme")) as Theme | null;
    const initial: Theme = saved ?? (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    setThemeState(initial);
    document.documentElement.setAttribute("data-theme", initial);
  }, []);

  const themeConfig = useMemo(() => themes[theme], [theme]);

  // Apply variables + persistence
  useEffect(() => {
    const root = document.documentElement;
    // set CSS vars
    (Object.entries(themeConfig) as [keyof ThemeConfig, string | ThemeConfig["gradient"]][]).forEach(([key, value]) => {
      if (key === "gradient") return;
      root.style.setProperty(`--${key}`, String(value));
    });
    // gradient as vars for CSS usage
    if (themeConfig.gradient) {
      root.style.setProperty("--gradient-from", themeConfig.gradient.from);
      root.style.setProperty("--gradient-to", themeConfig.gradient.to);
    }
    // set data-attr and persist
    root.setAttribute("data-theme", theme);
    try { localStorage.setItem("uiotas-theme", theme); } catch {}
    // smooth transition
    root.classList.add("theme-transition");
    return () => root.classList.remove("theme-transition");
  }, [theme, themeConfig]);

  const toggleTheme = () => setThemeState((t) => (t === "dark" ? "light" : "dark"));
  const setTheme = (t: Theme) => setThemeState(t);

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme, setTheme, themeConfig }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used within a ThemeProvider");
  return ctx;
}

// HOC: injects themeConfig prop into wrapped component
export function withTheme<P>(Wrapped: React.ComponentType<P & { themeConfig: ThemeConfig }>) {
  return function WithTheme(props: P) {
    const { themeConfig } = useTheme();
    return <Wrapped {...props} themeConfig={themeConfig} />;
  };
}

// Simple icon button toggle (kept for backwards-compat)
export function ThemeToggle() {
  const { theme, toggleTheme } = useTheme();
  const isDark = theme === "dark";
  return (
    <button type="button" onClick={toggleTheme} aria-label={`Switch to ${isDark ? "light" : "dark"} mode`} className="theme-toggle" style={{ background: "transparent", border: "none", cursor: "pointer" }}>
      <span aria-hidden className="icon" style={{ display: "inline-flex", width: 16, height: 16 }}>
        {isDark ? (
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="5"/>
            <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>
          </svg>
        ) : (
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/>
          </svg>
        )}
      </span>
      <span style={{ marginLeft: 8 }}>{isDark ? "Light" : "Dark"}</span>
    </button>
  );
}

// Advanced track toggle with styled-jsx
export function AdvancedThemeToggle() {
  const { theme, toggleTheme } = useTheme();
  return (
    <div className="theme-toggle-container">
      <button onClick={toggleTheme} className={`theme-toggle ${theme}-mode`} aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}>
        <div className="toggle-indicator">
          <span className="dark-icon">🌙</span>
          <span className="light-icon">☀️</span>
        </div>
      </button>
      <style jsx>{`
        .theme-toggle-container { position: relative; display: inline-block; }
        .theme-toggle { background: var(--panel); border: 1px solid var(--muted); border-radius: 20px; width: 60px; height: 30px; cursor: pointer; position: relative; transition: all .3s ease; overflow: hidden; }
        .toggle-indicator { position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: flex; align-items: center; transition: transform .3s ease; }
        .dark-icon, .light-icon { width: 50%; height: 100%; display: flex; align-items: center; justify-content: center; transition: opacity .3s ease; }
        .theme-toggle.dark-mode .toggle-indicator { transform: translateX(0); }
        .theme-toggle.light-mode .toggle-indicator { transform: translateX(100%); }
        .theme-toggle.dark-mode .dark-icon, .theme-toggle.light-mode .light-icon { opacity: 1; }
        .theme-toggle.dark-mode .light-icon, .theme-toggle.light-mode .dark-icon { opacity: 0; }
      `}</style>
    </div>
  );
}

// Pure color adjust utility
export function adjustColor(color: string, amount: number): string {
  const hex = color.replace('#', '');
  const num = parseInt(hex, 16);
  const amt = Math.round(2.55 * amount);
  let r = (num >> 16) + amt;
  let g = ((num >> 8) & 0x00ff) + amt;
  let b = (num & 0x0000ff) + amt;
  r = r < 0 ? 0 : r > 255 ? 255 : r;
  g = g < 0 ? 0 : g > 255 ? 255 : g;
  b = b < 0 ? 0 : b > 255 ? 255 : b;
  return `#${(1 << 24 | (r << 16) | (g << 8) | b).toString(16).slice(1)}`;
}

export default ThemeContext;
