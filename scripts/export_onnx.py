import json, numpy as np, torch, onnxruntime as ort
from common.models import *; from common.train_utils import load_ae
from scripts.t3_moe import build

def export(model, dummy, path, inputs, outputs, test_dummy=None):
    model = model.eval().cpu()
    dyn = {n: {0: "batch"} for n in inputs + outputs}
    kw = dict(input_names=inputs, output_names=outputs, dynamic_axes=dyn, opset_version=17)
    try: torch.onnx.export(model, dummy, path, dynamo=False, **kw)
    except TypeError: torch.onnx.export(model, dummy, path, **kw)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    d = test_dummy or dummy                                     # different batch size tests dynamic axes
    ref = model(*d); ref = ref if isinstance(ref, tuple) else (ref,)
    out = sess.run(None, {n: t.numpy() for n, t in zip(inputs, d)})
    diffs = [float(np.abs(r.detach().numpy() - o).max()) for r, o in zip(ref, out)]
    print(path, "max abs diff:", diffs); assert max(diffs) < 1e-4
    return dict(model=path, max_abs_diff=diffs)

class MoEOut(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x): y, w, _ = s.m(x); return y, w

if __name__ == "__main__":
    x1, x3 = torch.rand(1, 3, 128, 128), torch.rand(3, 3, 128, 128); log = []
    log.append(export(load_ae("checkpoints/t1_universal.pt").cpu(), (x1,), "models/universal.onnx", ["x"], ["y"], (x3,)))
    c = torch.load("checkpoints/t2_clf.pt", map_location="cpu"); clf = Classifier(c["cfg"]["base"], c["cfg"]["dropout"]); clf.load_state_dict(c["state"])
    log.append(export(clf, (x1,), "models/classifier.onnx", ["x"], ["logits"], (x3,)))
    for k in ["salt", "blur", "occlusion"]:
        log.append(export(load_ae(f"checkpoints/t2_{k}.pt").cpu(), (x1,), f"models/spec_{k}.onnx", ["x"], ["y"], (x3,)))
    moe = build(1.0).cpu(); ck = torch.load("checkpoints/t3_moe.pt", map_location="cpu")
    moe.T = ck["cfg"]["T"]; moe.load_state_dict(ck["state"])
    log.append(export(MoEOut(moe), (x1,), "models/soft_moe.onnx", ["x"], ["y", "w"], (x3,)))
    g = torch.load("checkpoints/t4_gen.pt", map_location="cpu"); cf = g["cfg"]
    G = UNetG(cf["base"], cf["emb"], cf["dropout"]); G.load_state_dict(g["state"])
    log.append(export(G, (x1, torch.tensor([0])), "models/generator.onnx", ["photo", "style"], ["sketch"],
                      (torch.rand(3, 3, 128, 128) * 2 - 1, torch.tensor([0, 1, 2]))))
    json.dump(log, open("models/onnx_verification.json", "w"), indent=2)