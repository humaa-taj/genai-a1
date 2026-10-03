import json, mlflow, torch, torch.nn.functional as F, optuna
from pytorch_msssim import ssim
from common.data import *; from common.models import *; from common.train_utils import *

tr, va, _ = load_split(); vm = json.load(open("data/val_manifest.json"))

def build(T):
    c = torch.load("checkpoints/t2_clf.pt", map_location=dev)
    gate = Classifier(c["cfg"]["base"], c["cfg"]["dropout"]); gate.load_state_dict(c["state"])
    experts = [load_ae(f"checkpoints/t2_{k}.pt") for k in ["salt", "blur", "occlusion"]]
    return SoftMoE(gate, experts, T).to(dev)

def joint_loss(y, w, logits, p, lab, cfg):
    l1 = F.l1_loss(p, y); s = 1 - ssim(p, y, data_range=1.0)
    ce = F.cross_entropy(logits, lab)
    bal = ((w.mean(0) - 0.25) ** 2).sum()
    return cfg["alpha"] * l1 + (1 - cfg["alpha"]) * s + cfg["wc"] * ce + cfg["wb"] * bal

@torch.no_grad()
def val(moe, vl):
    moe.eval(); l1 = s = n = 0; W = []
    for x, y, *_ in vl:
        x, y = x.to(dev), y.to(dev); p, w, _ = moe(x); b = len(x)
        l1 += F.l1_loss(p, y).item() * b; s += ssim(p, y, data_range=1.0).item() * b; n += b; W.append(w.cpu())
    mw = torch.cat(W).mean(0)
    return dict(l1=l1 / n, ssim=s / n, obj=l1 / n + 1 - s / n, min_w=mw.min().item(), max_w=mw.max().item())

def run(cfg, warm, joint, trial=None, save=None):
    moe = build(cfg["T"]); tl, vl = make_loaders(tr, va, vm, cfg["bs"], balanced=True); best = 1e9
    for phase, epochs in (("warmup", warm), ("joint", joint)):
        for p in moe.experts.parameters(): p.requires_grad = (phase == "joint")
        groups = [dict(params=moe.gate.parameters(), lr=cfg["lr_gate"])]
        if phase == "joint": groups.append(dict(params=moe.experts.parameters(), lr=cfg["lr_joint"]))
        opt = torch.optim.Adam(groups)
        for ep in range(epochs):
            moe.train()
            if phase == "warmup": moe.experts.eval()      # keep frozen experts' BN stats fixed
            for x, y, l, _ in tl:
                x, y, l = x.to(dev), y.to(dev), l.to(dev); p, w, lg = moe(x)
                loss = joint_loss(y, w, lg, p, l, cfg); opt.zero_grad(); loss.backward(); opt.step()
            r = val(moe, vl); step = ep + (warm if phase == "joint" else 0)
            mlflow.log_metrics({f"val_{k}": v for k, v in r.items()} | {"train_loss": loss.item()}, step=step)
            if phase == "joint":
                if trial:
                    trial.report(r["obj"], ep)
                    if r["min_w"] < 0.02 or trial.should_prune(): raise optuna.TrialPruned()   # routing collapse
                if r["obj"] < best:
                    best = r["obj"]
                    if save: torch.save(dict(cfg=cfg, state=moe.state_dict()), save)
    return best

def objective(t):
    cfg = dict(lr_gate=1e-4, lr_joint=t.suggest_float("lr_joint", 1e-5, 3e-4, log=True), bs=64,
               T=t.suggest_float("T", 0.5, 2.0), wc=t.suggest_float("wc", 0.01, 0.5, log=True),
               wb=t.suggest_float("wb", 1e-3, 0.1, log=True), alpha=t.suggest_float("alpha", 0.5, 0.95))
    with mlflow.start_run(run_name=f"trial_{t.number}", nested=True):
        mlflow.log_params(t.params); return run(cfg, 2, 6, t)

if __name__ == "__main__":
    s = run_study("task3_soft_moe", objective, 20)
    best = {**s.best_params, "lr_gate": 1e-4, "bs": 64}
    mlflow.set_experiment("task3_soft_moe")
    with mlflow.start_run(run_name="final"):
        mlflow.log_params(best); run(best, 5, 40, save="checkpoints/t3_moe.pt")