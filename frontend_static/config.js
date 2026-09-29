// Where the API lives. Served by the backend itself -> same origin (''); hosted
// separately (Vercel / custom domain) -> the Hugging Face Space running the backend.
(function () {
  var h = location.hostname;
  var separate = h.endsWith('.vercel.app') || h === 'rakshai.ronakbuilds.tech';
  window.RAKSHAI_API_ORIGIN = window.RAKSHAI_API_ORIGIN ||
    (separate ? 'https://fourtysevencode-rakshai.hf.space' : '');
})();
