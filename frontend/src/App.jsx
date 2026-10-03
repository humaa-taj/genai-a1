import { useState, useRef } from "react";

const TABS = [
  { id: "universal", name: "Universal Restoration", url: "/api/universal" },
  { id: "hard", name: "Hard-Routed Restoration", url: "/api/hard" },
  { id: "soft", name: "Soft Mixture-of-Experts Restoration", url: "/api/soft" },
  { id: "sketch", name: "Face-to-Sketch Generator", url: "/api/sketch" },
];

function Bars({ data }) {
  return Object.entries(data).map(([k, v]) => (
    <div key={k} className="my-1">
      <div className="flex justify-between text-sm">
        <span>{k}</span>
        <span>{(v * 100).toFixed(1)}%</span>
      </div>
      <div className="h-2 bg-slate-700 rounded">
        <div
          className="h-2 bg-sky-400 rounded"
          style={{ width: `${v * 100}%` }}
        />
      </div>
    </div>
  ));
}

function Workspace({ tab }) {
  const [file, setFile] = useState(null);
  const [corr, setCorr] = useState("none"),
    [level, setLevel] = useState(1),
    [style, setStyle] = useState(1);
  const [res, setRes] = useState(null),
    [busy, setBusy] = useState(false),
    [err, setErr] = useState("");
  const video = useRef(null);
  const isSketch = tab.id === "sketch";

  async function run() {
    if (!file) return setErr("Choose an image first");
    setBusy(true);
    setErr("");
    const fd = new FormData();
    fd.append("file", file);
    if (isSketch) fd.append("style", style);
    else {
      fd.append("corruption", corr);
      fd.append("level", level);
    }
    try {
      const r = await fetch(tab.url, { method: "POST", body: fd });
      if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
      setRes(await r.json());
    } catch (e) {
      setErr(String(e.message || e));
    }
    setBusy(false);
  }

  async function webcam() {
    const s = await navigator.mediaDevices.getUserMedia({ video: true });
    const v = video.current;
    v.srcObject = s;
    await v.play();
    const c = document.createElement("canvas");
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    c.getContext("2d").drawImage(v, 0, 0);
    s.getTracks().forEach((t) => t.stop());
    c.toBlob(
      (b) => setFile(new File([b], "cam.png", { type: "image/png" })),
      "image/png",
    );
  }

  return (
    <div className="space-y-4">
      <input
        type="file"
        accept="image/*"
        onChange={(e) => setFile(e.target.files[0])}
      />
      {isSketch ? (
        <>
          <select
            value={style}
            onChange={(e) => setStyle(+e.target.value)}
            className="bg-slate-800 p-2 rounded"
          >
            {[1, 2, 3].map((s) => (
              <option key={s} value={s}>
                Style {s}
              </option>
            ))}
          </select>
          <button
            onClick={webcam}
            className="px-3 py-2 bg-slate-700 rounded ml-2"
          >
            Capture webcam
          </button>
          <video ref={video} className="hidden" />
        </>
      ) : (
        <div className="flex gap-2">
          <select
            value={corr}
            onChange={(e) => setCorr(e.target.value)}
            className="bg-slate-800 p-2 rounded"
          >
            <option value="none">
              No corruption (already corrupted / clean)
            </option>
            <option value="salt">Salt & pepper</option>
            <option value="blur">Gaussian blur</option>
            <option value="occlusion">Occlusion</option>
          </select>
          <select
            value={level}
            onChange={(e) => setLevel(+e.target.value)}
            className="bg-slate-800 p-2 rounded"
          >
            <option value={0}>Low</option>
            <option value={1}>Medium</option>
            <option value={2}>High</option>
          </select>
        </div>
      )}
      <button
        onClick={run}
        disabled={busy}
        className="px-4 py-2 bg-sky-500 rounded font-semibold"
      >
        {busy ? "Running…" : "Run"}
      </button>
      {err && <p className="text-red-400">{err}</p>}
      {res && (
        <div className="grid md:grid-cols-2 gap-4">
          <figure>
            <img src={res.input} className="w-full rounded" />
            <figcaption>Input</figcaption>
          </figure>
          <figure>
            <img src={res.output} className="w-full rounded" />
            <figcaption>Output</figcaption>
            <a
              href={res.output}
              download="result.png"
              className="text-sky-400 text-sm"
            >
              Download
            </a>
          </figure>
          <div className="md:col-span-2 text-sm space-y-2">
            <p>
              Inference: <b>{res.inference_ms} ms</b>
              {res.corruption &&
                ` · corruption: ${JSON.stringify(res.corruption)}`}
            </p>
            {res.probs && (
              <>
                <p>
                  Predicted: <b>{res.predicted}</b> → expert:{" "}
                  <b>{res.expert}</b>
                </p>
                <Bars data={res.probs} />
              </>
            )}
            {res.weights && (
              <>
                <p>
                  Dominant expert: <b>{res.dominant}</b>
                </p>
                <Bars data={res.weights} />
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [i, setI] = useState(0);
  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex">
      <nav className="w-64 p-4 space-y-2 bg-slate-950">
        {TABS.map((t, k) => (
          <button
            key={t.id}
            onClick={() => setI(k)}
            className={`block w-full text-left p-2 rounded ${i === k ? "bg-sky-600" : "hover:bg-slate-800"}`}
          >
            {t.name}
          </button>
        ))}
      </nav>
      <main className="flex-1 p-6">
        <h1 className="text-2xl mb-4">{TABS[i].name}</h1>
        <Workspace key={TABS[i].id} tab={TABS[i]} />
      </main>
    </div>
  );
}
