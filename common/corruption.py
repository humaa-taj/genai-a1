import numpy as np, cv2
CLASSES = ["clean", "salt", "blur", "occlusion"]
SIZE = 128
LEVELS = ["low", "medium", "high"]

def salt_pepper(img, p, seed):
    rng = np.random.default_rng(seed)
    out = img.copy()
    hit = rng.random(img.shape[:2]) < p
    white = rng.random(img.shape[:2]) < 0.5
    out[hit & white] = 255
    out[hit & ~white] = 0
    return out

def gaussian_blur(img, k, sigma):
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma)

def make_rects(n, frac, seed, size=SIZE):
    rng = np.random.default_rng(seed)
    areas = rng.dirichlet(np.ones(n)) * frac * size * size
    rects = []
    for a in areas:
        ar = rng.uniform(0.5, 2.0)
        w = int(min(size, max(2, round(np.sqrt(a * ar)))))
        h = int(min(size, max(2, round(a / w))))
        x = int(rng.integers(0, size - w + 1)); y = int(rng.integers(0, size - h + 1))
        rects.append([x, y, w, h])
    return rects

def occlude(img, rects):
    out = img.copy()
    for x, y, w, h in rects:
        out[y:y + h, x:x + w] = 0
    return out

def apply_spec(img, spec):
    t = spec["type"]
    if t == "salt": return salt_pepper(img, spec["p"], spec["seed"])
    if t == "blur": return gaussian_blur(img, spec["k"], spec["sigma"])
    if t == "occlusion": return occlude(img, spec["rects"])
    return img

def sample_spec(kind, rng):
    """Training / validation sampling ranges from the assignment."""
    seed = int(rng.integers(1 << 31))
    if kind == "salt":
        return dict(type="salt", p=float(rng.uniform(.02, .15)), seed=seed)
    if kind == "blur":
        return dict(type="blur", k=int(rng.choice([3, 5, 7])), sigma=float(rng.uniform(.5, 2.5)))
    if kind == "occlusion":
        n = int(rng.integers(1, 4)); f = float(rng.uniform(.10, .35))
        return dict(type="occlusion", rects=make_rects(n, f, seed), frac=f, seed=seed)
    return dict(type="clean")

def fixed_spec(kind, li, seed):
    """Fixed test severities (li = 0,1,2)."""
    if kind == "salt":
        return dict(type="salt", p=[.03, .08, .15][li], seed=seed)
    if kind == "blur":
        k, s = [(3, .7), (5, 1.5), (7, 2.5)][li]
        return dict(type="blur", k=k, sigma=s)
    n, f = [(1, .10), (2, .20), (3, .35)][li]
    return dict(type="occlusion", rects=make_rects(n, f, seed), frac=f, seed=seed)

def severity_bin(spec):
    t = spec["type"]
    if t == "clean": return "none"
    v, cuts = {"salt": (spec.get("p"), (.0633, .1067)),
               "blur": (spec.get("sigma"), (1.167, 1.833)),
               "occlusion": (spec.get("frac"), (.183, .267))}[t]
    return LEVELS[0] if v < cuts[0] else LEVELS[1] if v < cuts[1] else LEVELS[2]

def build_val_manifest(n, seed=42):
    rng = np.random.default_rng(seed)
    kinds = np.resize(np.arange(4), n); rng.shuffle(kinds)   # balanced
    out = []
    for i, k in enumerate(kinds):
        spec = sample_spec(CLASSES[k], rng)
        out.append(dict(idx=i, kind=CLASSES[k], severity=severity_bin(spec), spec=spec))
    return out

def build_test_manifest(n):
    out = []
    for i in range(n):
        out.append(dict(idx=i, kind="clean", severity="none", spec=dict(type="clean")))
        for ci, kind in enumerate(["salt", "blur", "occlusion"]):
            for li in range(3):
                spec = fixed_spec(kind, li, 100000 + i * 100 + ci * 10 + li)
                out.append(dict(idx=i, kind=kind, severity=LEVELS[li], spec=spec))
    return out