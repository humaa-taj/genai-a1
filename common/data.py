import json, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from .corruptions import *

def to_t(a): return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float() / 255

def load_split(R="data"):
    tv = np.load(f"{R}/trainval.npy"); sp = json.load(open(f"{R}/split.json"))
    return tv[sp["train"]], tv[sp["val"]], np.load(f"{R}/test.npy")

class TrainDS(Dataset):
    """Runtime corruption. `kinds` limits which conditions are sampled (specialists)."""
    def __init__(self, imgs, kinds=CLASSES): self.imgs, self.kinds = imgs, list(kinds)
    def __len__(self): return len(self.imgs)
    def __getitem__(self, item):
        i, lab = item if isinstance(item, tuple) else (item, None)
        rng = np.random.default_rng()
        kind = CLASSES[lab] if lab is not None else self.kinds[int(rng.integers(len(self.kinds)))]
        c = self.imgs[i]; x = apply_spec(c, sample_spec(kind, rng))
        return to_t(x), to_t(c), CLASSES.index(kind), i

class ManifestDS(Dataset):
    """Deterministic validation/test corruption from a stored manifest."""
    def __init__(self, imgs, manifest, kinds=None):
        self.imgs = imgs
        self.m = [e for e in manifest if kinds is None or e["kind"] in kinds]
    def __len__(self): return len(self.m)
    def __getitem__(self, k):
        e = self.m[k]; c = self.imgs[e["idx"]]
        return to_t(apply_spec(c, e["spec"])), to_t(c), CLASSES.index(e["kind"]), k

class BalancedSampler:
    """Every batch has an equal number of each requested condition."""
    def __init__(self, n, bs, kinds=(0, 1, 2, 3)): self.n, self.bs, self.kinds = n, bs, kinds
    def __len__(self): return self.n // self.bs
    def __iter__(self):
        for _ in range(len(self)):
            labs = np.resize(self.kinds, self.bs); np.random.shuffle(labs)
            yield [(int(np.random.randint(self.n)), int(l)) for l in labs]

def make_loaders(train, val, manifest, bs, kinds=CLASSES, balanced=False, workers=4):
    ds = TrainDS(train, kinds)
    if balanced:
        tl = DataLoader(ds, batch_sampler=BalancedSampler(len(train), bs, [CLASSES.index(k) for k in kinds]),
                        num_workers=workers)
    else:
        tl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=workers, drop_last=True)
    vl = DataLoader(ManifestDS(val, manifest, kinds), batch_size=128, num_workers=2)
    return tl, vl