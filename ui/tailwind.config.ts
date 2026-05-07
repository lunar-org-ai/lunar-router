import type { Config } from 'tailwindcss';

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Geist', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['Geist Mono', 'ui-monospace', 'Cascadia Code', 'Source Code Pro', 'monospace'],
      },
      colors: {
        bg: 'var(--bg)',
        'bg-elev': 'var(--bg-elev)',
        'bg-muted': 'var(--bg-muted)',
        'bg-sunken': 'var(--bg-sunken)',
        fg: 'var(--fg)',
        'fg-muted': 'var(--fg-muted)',
        'fg-subtle': 'var(--fg-subtle)',
        border: 'var(--border)',
        'border-strong': 'var(--border-strong)',
        accent: 'var(--accent)',
        'accent-soft': 'var(--accent-soft)',
        'accent-fg': 'var(--accent-fg)',
        warn: 'var(--warn)',
        'warn-soft': 'var(--warn-soft)',
        'warn-fg': 'var(--warn-fg)',
        bad: 'var(--bad)',
        'bad-soft': 'var(--bad-soft)',
        'bad-fg': 'var(--bad-fg)',
        info: 'var(--info)',
        'info-soft': 'var(--info-soft)',
        'info-fg': 'var(--info-fg)',
      },
      borderRadius: {
        DEFAULT: 'var(--radius)',
        sm: 'var(--radius-sm)',
        lg: 'var(--radius-lg)',
      },
      boxShadow: {
        sm: 'var(--shadow-sm)',
        DEFAULT: 'var(--shadow)',
        lg: 'var(--shadow-lg)',
      },
    },
  },
  plugins: [],
} satisfies Config;
