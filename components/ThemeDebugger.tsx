"use client";

import React, { useState } from "react";
import { useTheme } from "@/components/ThemeUtils";

export function ThemeDebugger() {
  const [isOpen, setIsOpen] = useState(false);
  const { theme, setTheme, themeConfig } = useTheme();

  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        style={{
          position: "fixed",
          bottom: "20px",
          right: "20px",
          background: "var(--panel)",
          border: "1px solid var(--muted)",
          padding: "10px",
          borderRadius: "8px",
          zIndex: 100,
          color: "var(--text)",
        }}
      >
        🔧 Theme Debug
      </button>
    );
  }

  return (
    <div
      style={{
        position: "fixed",
        bottom: "20px",
        right: "20px",
        background: "var(--panel)",
        border: "1px solid var(--muted)",
        padding: "20px",
        borderRadius: "12px",
        zIndex: 100,
        maxWidth: "400px",
        boxShadow: "0 4px 6px rgba(0,0,0,0.1)",
        color: "var(--text)",
      }}
    >
      <h3>Theme Debugger</h3>

      <div>
        <h4>Current Theme: {theme}</h4>
        <div style={{ display: "flex", gap: "10px", marginBottom: "15px" }}>
          <button
            onClick={() => setTheme("dark")}
            style={{
              background: theme === "dark" ? "var(--accent)" : "var(--panel)",
              color: "var(--text)",
              border: "1px solid var(--muted)",
              padding: "5px 10px",
              borderRadius: "6px",
            }}
          >
            Dark Mode
          </button>
          <button
            onClick={() => setTheme("light")}
            style={{
              background: theme === "light" ? "var(--accent)" : "var(--panel)",
              color: "var(--text)",
              border: "1px solid var(--muted)",
              padding: "5px 10px",
              borderRadius: "6px",
            }}
          >
            Light Mode
          </button>
        </div>
      </div>

      <div>
        <h4>Theme Variables</h4>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: "10px",
            marginBottom: "15px",
          }}
        >
          {Object.entries(themeConfig).map(([key, value]) => {
            if (key === "gradient") return null;
            return (
              <div key={key} style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                <div
                  style={{
                    width: "20px",
                    height: "20px",
                    background: value as string,
                    border: "1px solid var(--muted)",
                  }}
                />
                <span>
                  {key}: {value as string}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      <button
        onClick={() => setIsOpen(false)}
        style={{
          background: "var(--panel)",
          color: "var(--text)",
          border: "1px solid var(--muted)",
          padding: "5px 10px",
          borderRadius: "6px",
          marginTop: "10px",
        }}
      >
        Close
      </button>
    </div>
  );
}

export default ThemeDebugger;

