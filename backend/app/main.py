import base64, io, os, time
import numpy as np, onnxruntime as ort
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from corruptions import CLASSES, apply_spec, fixed_spec

MD = os.getenv("MODEL_DIR", "/models"); MAXB = 10 * 1024 * 1024
S = {}
def sess(n): return ort.InferenceSession(f"{MD}/{n}.onnx", providers=["CPUExecutionProvider"])

app = FastAPI(title="GenAI A1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
def load():
    for n in ["universal", "classifier", "spec_salt", "spec_blur", "spec_occlusion", "soft_moe", "generator"]:
        try: S[n] = sess(n)
        except Exception as e: print("missing", n, e)

@app.get("/api/health")
def health(): return {"status": "ok", "models": sorted(S)}

async def read_img(f: UploadFile):
    if f.content_type not in ("image/png", "image/jpeg", "image/webp"): raise HTTPException(415, "PNG/JPEG/WebP only")
    b = await f.read()
    if len(b) > MAXB: raise HTTPException(413, "File too large (10 MB max)")
    try: im = Image.open(io.BytesIO(b)); im.verify(); im = Image.open(io.BytesIO(b))
    except Exception: raise HTTPException(400, "Not a valid image")
    return np.asarray(im.convert("RGB").resize((128, 128), Image.BICUBIC))

def b64(a):
    buf = io.BytesIO(); Image.fromarray(a).save(buf, "PNG"); return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def corrupt(img, kind, level):
    if kind in (None, "", "none", "clean"): return img, {"type": "none"}
    spec = fixed_spec(kind, int(level), seed=42); spec.pop("rects", None) if False else None
    return apply_spec(img, spec), spec

def tin(a): return (a.astype(np.float32) / 255).transpose(2, 0, 1)[None]
def tout(t): return (np.clip(t[0].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
def softmax(z): e = np.exp(z - z.max()); return e / e.sum()

@app.post("/api/universal")
async def universal(file: UploadFile = File(...), corruption: str = Form("none"), level: int = Form(1)):
    img = await read_img(file); x, spec = corrupt(img, corruption, level)
    t = time.perf_counter(); y = S["universal"].run(None, {"x": tin(x)})[0]; ms = (time.perf_counter() - t) * 1000
    return {"input": b64(x), "output": b64(tout(y)), "corruption": spec, "inference_ms": round(ms, 2)}

@app.post("/api/hard")
async def hard(file: UploadFile = File(...), corruption: str = Form("none"), level: int = Form(1)):
    img = await read_img(file); x, spec = corrupt(img, corruption, level); xi = tin(x)
    t = time.perf_counter()
    p = softmax(S["classifier"].run(None, {"x": xi})[0][0]); k = int(p.argmax())
    expert = ["identity (bypass)", "salt", "blur", "occlusion"][k]
    y = xi if k == 0 else S[f"spec_{CLASSES[k] if k != 1 else 'salt'}"].run(None, {"x": xi})[0]
    ms = (time.perf_counter() - t) * 1000
    return {"input": b64(x), "output": b64(tout(y)), "probs": dict(zip(CLASSES, map(float, p))),
            "predicted": CLASSES[k], "expert": expert, "corruption": spec, "inference_ms": round(ms, 2)}

@app.post("/api/soft")
async def soft(file: UploadFile = File(...), corruption: str = Form("none"), level: int = Form(1)):
    img = await read_img(file); x, spec = corrupt(img, corruption, level)
    t = time.perf_counter(); y, w = S["soft_moe"].run(None, {"x": tin(x)}); ms = (time.perf_counter() - t) * 1000
    names = ["identity", "salt", "blur", "occlusion"]; w = w[0].tolist()
    return {"input": b64(x), "output": b64(tout(y)), "weights": dict(zip(names, w)),
            "dominant": names[int(np.argmax(w))], "corruption": spec, "inference_ms": round(ms, 2)}

@app.post("/api/sketch")
async def sketch(file: UploadFile = File(...), style: int = Form(1)):
    if style not in (1, 2, 3): raise HTTPException(422, "style must be 1, 2 or 3")
    img = await read_img(file); x = (img.astype(np.float32) / 127.5 - 1).transpose(2, 0, 1)[None]
    t = time.perf_counter(); y = S["generator"].run(None, {"photo": x, "style": np.array([style - 1], dtype=np.int64)})[0]
    ms = (time.perf_counter() - t) * 1000
    out = ((np.clip(y[0].transpose(1, 2, 0), -1, 1) + 1) * 127.5).astype(np.uint8)
    return {"input": b64(img), "output": b64(out), "style": style, "inference_ms": round(ms, 2)}