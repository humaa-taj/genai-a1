import collections, numpy as np, torch
from torch.utils.data import DataLoader
from pytorch_msssim import ssim
from common.data import ManifestDS

@torch.no_grad()
def eval_fn(restore, imgs, manifest, dev, bs=128):
    ds = ManifestDS(imgs, manifest); rows = collections.defaultdict(lambda: collections.defaultdict(list))
    for x, y, l, k in DataLoader(ds, batch_size=bs):
        x, y = x.to(dev), y.to(dev); p = restore(x, l.to(dev)); p = p[0] if isinstance(p, tuple) else p
        l1 = (p - y).abs().mean((1, 2, 3)); psnr = 10 * torch.log10(1 / ((p - y) ** 2).mean((1, 2, 3)).clamp_min(1e-10))
        s = ssim(p, y, data_range=1.0, size_average=False)
        for j, idx in enumerate(k.tolist()):
            e = ds.m[idx]
            for n, v in zip(("l1", "psnr", "ssim"), (l1[j], psnr[j], s[j])): rows[(e["kind"], e["severity"])][n].append(v.item())
    return {f"{a}|{b}": {n: float(np.mean(v)) for n, v in d.items()} for (a, b), d in rows.items()}