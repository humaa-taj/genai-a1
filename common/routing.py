import torch
from .models import Classifier
from .train_utils import load_ae, dev

SPEC_KINDS = ["salt", "blur", "occlusion"]   # class indices 1, 2, 3 (0 = clean)

def load_clf(path="checkpoints/t2_clf.pt"):
    c = torch.load(path, map_location=dev); cfg = c["cfg"]
    m = Classifier(cfg["base"], cfg["dropout"]).to(dev)
    m.load_state_dict(c["state"]); return m.eval()

def load_specialists(d="checkpoints"):
    return {k: load_ae(f"{d}/t2_{k}.pt") for k in SPEC_KINDS}

@torch.no_grad()
def hard_route(x, clf, specs, labels=None):
    """labels given -> oracle routing; labels None -> predicted routing.
    Class 0 (clean) is an identity bypass. Returns (restored, probs, route)."""
    probs = clf(x).softmax(1)
    route = labels.to(x.device) if labels is not None else probs.argmax(1)
    out = x.clone()
    for j, k in enumerate(SPEC_KINDS, start=1):
        m = route == j
        if m.any(): out[m] = specs[k](x[m])
    return out, probs, route