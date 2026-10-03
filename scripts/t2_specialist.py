import json, mlflow
from common.data import *; from common.train_utils import *

tr, va, _ = load_split(); vm = json.load(open("data/val_manifest.json"))
SPEC = ["salt", "blur", "occlusion"]

def objective(t):
    cfg = dict(lr=t.suggest_float("lr", 1e-4, 3e-3, log=True), bs=t.suggest_categorical("bs", [32, 64, 128]),
                latent=t.suggest_categorical("latent", [128, 256, 512]),
                base=t.suggest_categorical("base", [16, 32, 48]),
                dropout=t.suggest_float("dropout", 0.0, 0.3),
               alpha=t.suggest_float("alpha", 0.5, 0.95))
    with mlflow.start_run(run_name=f"trial_{t.number}", nested=True):
        mlflow.log_params(t.params)
        tl, vl = make_loaders(tr, va, vm, cfg["bs"], kinds=SPEC)
        return train_ae(cfg, tl, vl, 10, trial=t)

if __name__ == "__main__":
    s = run_study("task2_specialists_shared", objective, 25); best = {**s.best_params, "dropout": 0.1}
    mlflow.set_experiment("task2_specialists")
    for k in SPEC:                                  # independent parameters, own corruption only
        with mlflow.start_run(run_name=f"specialist_{k}"):
            mlflow.log_params(best)
            tl, vl = make_loaders(tr, va, vm, best["bs"], kinds=[k])
            train_ae(best, tl, vl, 80, save=f"checkpoints/t2_{k}.pt")