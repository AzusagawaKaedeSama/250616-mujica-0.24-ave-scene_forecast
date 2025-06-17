const colors = require('tailwindcss/colors');

module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class', // or 'media' or 'class'
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#DC2626', // red-600
          dark: '#B91C1C',    // red-700
        },
        // Removed secondary for now, can be re-added if needed
        neutral: {
          DEFAULT: '#737373', // neutral-500
          50: '#fafafa',
          100: '#f5f5f5',
          200: '#e5e5e5',
          300: '#d4d4d4',
          400: '#a3a3a3',
          600: '#525252',
          700: '#404040',
          800: '#262626',
          900: '#171717',
        },
        // gradient object can be removed if not used in new theme
        // gradient: {
        //   start: '#ef4444', 
        //   middle: '#dc2626',
        //   end: '#b91c1c'
        // }
      },
      fontFamily: {
        // Removed default 'sans' stack
        // Define specific font families for targeted use
        'womby': ['Womby', 'ui-sans-serif', 'system-ui', '-apple-system', 'BlinkMacSystemFont', "Segoe UI", 'Roboto', "Helvetica Neue", 'Arial', "Noto Sans", 'sans-serif', "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji"],
        'antYaHei': ['Ant YaHei', 'ui-sans-serif', 'system-ui', '-apple-system', 'sans-serif'], // For Chinese titles/bold
        'sourceHanBold': ['SourceHanSansSC-Bold', 'ui-sans-serif', 'system-ui', '-apple-system', 'sans-serif'] // For normal Chinese text
      },
    },
  },
  plugins: [],
} 