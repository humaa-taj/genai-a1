import torch, torch.nn.functional as F, mlflow, optuna
from pytorch_msssim import ssim
from .models import AE

dev = "cuda" if torch.cuda.is_available() else "cpu"

def l1ssim(p, y, a): return a * F.l1_loss(p, y) + (1 - a) * (1 - ssim(p, y, data_range=1.0))

@torch.no_grad()
def eval_ae(m, loader):
    m.eval(); l1 = s = n = 0
    for x, y, *_ in loader:
        x, y = x.to(dev), y.to(dev); p = m(x); b = len(x)
        l1 += F.l1_loss(p, y).item() * b; s += ssim(p, y, data_range=1.0).item() * b; n += b
    return dict(l1=l1 / n, ssim=s / n, obj=l1 / n + 1 - s / n)   # objective: L1 + (1 - SSIM)

def save_ae(m, cfg, path): torch.save(dict(cfg=cfg, state=m.state_dict()), path)

def load_ae(path):
    c = torch.load(path, map_location=dev); cfg = c["cfg"]
    m = AE(cfg["base"], cfg["latent"], cfg["dropout"]).to(dev); m.load_state_dict(c["state"]); return m.eval()

def train_ae(cfg, tl, vl, epochs, trial=None, save=None):
    m = AE(cfg["base"], cfg["latent"], cfg["dropout"]).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=cfg.get("wd", 1e-5))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs); best = 1e9
    for ep in range(epochs):
        m.train(); tot = 0
        for x, y, *_ in tl:
            x, y = x.to(dev), y.to(dev); loss = l1ssim(m(x), y, cfg["alpha"])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        sch.step(); r = eval_ae(m, vl)
        mlflow.log_metrics({"train_loss": tot / len(tl), **{f"val_{k}": v for k, v in r.items()}}, step=ep)
        if r["obj"] < best:
            best = r["obj"]
            if save: save_ae(m, cfg, save)
        if trial:
            trial.report(r["obj"], ep)
            if trial.should_prune(): raise optuna.TrialPruned()
    return best

def run_study(name, objective, n_trials, direction="minimize"):
    mlflow.set_tracking_uri("sqlite:///mlflow.db"); mlflow.set_experiment(name)
    study = optuna.create_study(study_name=name, storage="sqlite:///optuna.db", load_if_exists=True,
                                direction=direction, sampler=optuna.samplers.TPESampler(seed=42),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3))
    with mlflow.start_run(run_name=f"{name}_study"):
        study.optimize(objective, n_trials=n_trials)
    return study