import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { chineseHelp } from './helpLanguage.js';

test('every labelled parameter help badge has Chinese text', () => {
  const missing = [];
  for (const file of ['./components/FaceSwap.jsx', './components/Settings.jsx', './components/FaceManager.jsx']) {
    const source = readFileSync(new URL(file, import.meta.url), 'utf8');
    for (const control of source.matchAll(/<(?:Select|Slider|Toggle|TextInput)\b[\s\S]*?\/>/g)) {
      if (!/\binfo\s*=/.test(control[0])) continue;
      const label = control[0].match(/\blabel="([^"]+)"/)?.[1];
      if (label && !chineseHelp[label]) missing.push(`${file}: ${label}`);
    }
  }
  assert.deepEqual(missing, []);
});
