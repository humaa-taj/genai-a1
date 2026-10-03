import json, numpy as np
from PIL import Image
from torchvision.datasets import OxfordIIITPet
from common.corruptions import build_val_manifest, build_test_manifest
R = "data"
def load(split):
    ds = OxfordIIITPet(R, split=split, download=True)
    return np.stack([np.asarray(im.convert("RGB").resize((128, 128), Image.BICUBIC)) for im, _ in ds])
tv, te = load("trainval"), load("test")
np.save(f"{R}/trainval.npy", tv); np.save(f"{R}/test.npy", te)
perm = np.random.RandomState(42).permutation(len(tv)); k = int(.8 * len(tv))
json.dump({"train": perm[:k].tolist(), "val": perm[k:].tolist()}, open(f"{R}/split.json", "w"))
json.dump(build_val_manifest(len(tv) - k), open(f"{R}/val_manifest.json", "w"))
json.dump(build_test_manifest(len(te)), open(f"{R}/test_manifest.json", "w"))
print(len(tv), k, len(te))