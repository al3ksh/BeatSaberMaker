/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bs: {
          bg: "#09090b",
          surface: "#18181b",
          border: "#27272a",
          red: "#ef4444",
          blue: "#3b82f6",
        },
      },
    },
  },
  plugins: [],
};
