// Tiny Web-Audio "board feel" engine: a synthesized wood knock for moves and a
// sharper thud for captures. No audio assets (nothing to host or license), works
// offline, and is created lazily on the first user gesture (a drag) so it obeys
// browser autoplay rules.

let ctx: AudioContext | null = null;

function audio(): AudioContext | null {
  if (typeof window === "undefined") return null;
  if (!ctx) {
    const Ctor =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!Ctor) return null;
    ctx = new Ctor();
  }
  if (ctx.state === "suspended") void ctx.resume();
  return ctx;
}

/** A pitched "knock": a fast-decaying low body + a high-passed click transient. */
function knock(freq: number, dur: number, gain: number, clickAmt: number) {
  const c = audio();
  if (!c) return;
  const now = c.currentTime;

  const osc = c.createOscillator();
  osc.type = "sine";
  osc.frequency.setValueAtTime(freq, now);
  osc.frequency.exponentialRampToValueAtTime(Math.max(40, freq * 0.55), now + dur);

  const g = c.createGain();
  g.gain.setValueAtTime(0.0001, now);
  g.gain.exponentialRampToValueAtTime(gain, now + 0.004);
  g.gain.exponentialRampToValueAtTime(0.0001, now + dur);
  osc.connect(g).connect(c.destination);
  osc.start(now);
  osc.stop(now + dur + 0.02);

  // Click transient (short filtered noise): the "attack" of the knock.
  const len = Math.floor(c.sampleRate * 0.02);
  const buf = c.createBuffer(1, len, c.sampleRate);
  const data = buf.getChannelData(0);
  for (let i = 0; i < len; i++) data[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, 2);
  const noise = c.createBufferSource();
  noise.buffer = buf;
  const hp = c.createBiquadFilter();
  hp.type = "highpass";
  hp.frequency.value = 1800;
  const ng = c.createGain();
  ng.gain.value = clickAmt * gain;
  noise.connect(hp).connect(ng).connect(c.destination);
  noise.start(now);
}

export function playMove(): void {
  knock(205, 0.11, 0.22, 0.35);
}

export function playCapture(): void {
  knock(150, 0.16, 0.32, 0.6);
}
