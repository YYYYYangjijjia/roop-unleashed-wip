// A queued preview owns its payload and identity. Never rebuild either from
// the closure of the request that happened to be running before it.
export function createPreviewQueue({ isCurrent, run, commit, onError, onBusy }) {
  let busy = false;
  let pending = null;
  async function drain(task) {
    busy = true;
    onBusy(true);
    try {
      while (task) {
        if (isCurrent(task.key)) {
          try {
            const result = await run(task);
            if (result?.error) throw new Error(String(result.message || result.error));
            if (result && isCurrent(task.key)) commit(task, result);
          } catch (error) {
            // A failed service must not immediately receive another queued
            // render. A later explicit refresh can start a new request.
            pending = null;
            if (isCurrent(task.key)) onError(error);
            break;
          }
        }
        task = pending;
        pending = null;
      }
    } finally {
      busy = false;
      onBusy(false);
    }
  }
  return {
    enqueue(task) {
      if (busy) { pending = task; return; }
      return drain(task);
    },
    clearPending() { pending = null; },
  };
}

// Decode BOTH images before publishing a pair. Superseded completions and
// partial/failed pairs can never replace the last complete pair.
export function createPreviewPairLoader(loadImage, publish, onError = () => {}) {
  let generation = 0;
  return {
    async load(pair) {
      const ticket = ++generation;
      if (!pair.beforeSrc) { publish(pair); return; }
      try {
        await Promise.all([...new Set([pair.beforeSrc, pair.afterSrc])].map(loadImage));
        if (ticket === generation) publish(pair);
      } catch (error) {
        if (ticket === generation) onError(error);
      }
    },
    cancel() { generation += 1; },
  };
}
