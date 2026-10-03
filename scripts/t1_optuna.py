import json, mlflow
from common.data import load_split, make_loaders
from common.train_utils import *

tr, va, _ = load_split(); vm = json.load(open("data/val_manifest.json"))
TRIAL_EPOCHS, FINAL_EPOCHS, N_TRIALS = 10, 100, 30

def cfg_from(t):
    return dict(lr=t.suggest_float("lr", 1e-4, 3e-3, log=True), bs=t.suggest_categorical("bs", [32, 64, 128]),
                latent=t.suggest_categorical("latent", [128, 256, 512, 1024]),
                base=t.suggest_categorical("base", [16, 32, 48, 64]),
                dropout=t.suggest_float("dropout", 0.0, 0.3), alpha=t.suggest_float("alpha", 0.5, 0.95))

def objective(t):
    cfg = cfg_from(t)
    with mlflow.start_run(run_name=f"trial_{t.number}", nested=True):
        mlflow.log_params(t.params)
        tl, vl = make_loaders(tr, va, vm, cfg["bs"])
        return train_ae(cfg, tl, vl, TRIAL_EPOCHS, trial=t)

if __name__ == "__main__":
    study = run_study("task1_universal_ae", objective, N_TRIALS)
    best = {**study.best_params}; print(best)
    json.dump(best, open("checkpoints/t1_best.json", "w"))
    mlflow.set_experiment("task1_universal_ae")
    with mlflow.start_run(run_name="final"):
        mlflow.log_params(best)
        tl, vl = make_loaders(tr, va, vm, best["bs"])
        print(train_ae(best, tl, vl, FINAL_EPOCHS, save="checkpoints/t1_universal.pt"))
        mlflow.log_artifact("checkpoints/t1_universal.pt")