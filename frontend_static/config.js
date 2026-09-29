// Where the API lives. Served by the backend itself -> same origin (''); hosted on
// Vercel -> the Hugging Face Space that runs the backend.
window.RAKSHAI_API_ORIGIN = window.RAKSHAI_API_ORIGIN ||
  (location.hostname.endsWith('.vercel.app') ? 'https://fourtysevencode-rakshai.hf.space' : '');
