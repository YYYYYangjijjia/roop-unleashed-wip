export function validateSwapModelSupport(payload, meta) {
  if (payload?.swap_model !== 'alphaface' || payload.fake_preview === false) return;
  if (!meta?.capabilities?.alphaface || !meta?.swap_models?.includes('alphaface')) {
    throw new Error('AlphaFace is unavailable in this backend. Restart the backend and refresh the page. InSwapper will not be used as a fallback.');
  }
  if (payload.a_compatibility_mode) {
    throw new Error('AlphaFace requires Legacy rendering compatibility to be off on the main page. Average identity may stay on.');
  }
  if (meta.swap_model_status?.alphaface?.installed === false) {
    throw new Error('AlphaFace model or identity mapping is missing locally. No automatic download or model fallback will occur.');
  }
}
