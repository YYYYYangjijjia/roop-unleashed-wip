import { useEffect, useRef, useState } from 'react';
import { createPreviewPairLoader } from './previewSync';

function decodeImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
      if (image.decode) image.decode().then(resolve, reject);
      else resolve();
    };
    image.onerror = () => reject(new Error('Preview image could not be loaded'));
    image.src = src;
  });
}

export default function usePreviewPair(beforeSrc, afterSrc, key) {
  const [pair, setPair] = useState({ beforeSrc: '', afterSrc: '', key: '' });
  const loader = useRef(null);
  if (!loader.current) loader.current = createPreviewPairLoader(decodeImage, setPair);
  useEffect(() => {
    loader.current.load({ beforeSrc, afterSrc: afterSrc || beforeSrc, key });
    return () => loader.current.cancel();
  }, [beforeSrc, afterSrc, key]);
  return pair;
}
