import json, mlflow, torch, torch.nn.functional as F, optuna, numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from common.data import *; from common.models import Classifier; from common.train_utils import *

tr, va, te = load_split(); vm = json.load(open("data/val_manifest.json"))

@torch.no_grad()
def acc(m, vl):
    m.eval(); c = n = 0
    for x, y, l, _ in vl:
        c += (m(x.to(dev)).argmax(1).cpu() == l).sum().item(); n += len(l)
    return c / n

def fit(cfg, epochs, trial=None, save=None):
    tl, vl = make_loaders(tr, va, vm, cfg["bs"], balanced=True)
    m = Classifier(cfg["base"], cfg["dropout"]).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"]); best = 0
    for ep in range(epochs):
        m.train()
        for x, _, l, _ in tl:
            loss = F.cross_entropy(m(x.to(dev)), l.to(dev)); opt.zero_grad(); loss.backward(); opt.step()
        a = acc(m, vl); mlflow.log_metrics({"val_acc": a, "train_loss": loss.item()}, step=ep)
        if a > best:
            best = a
            if save: torch.save(dict(cfg=cfg, state=m.state_dict()), save)
        if trial:
            trial.report(a, ep)
            if trial.should_prune(): raise optuna.TrialPruned()
    return best

def objective(t):
    cfg = dict(lr=t.suggest_float("lr", 1e-4, 3e-3, log=True), bs=t.suggest_categorical("bs", [32, 64, 128]),
               base=t.suggest_categorical("base", [16, 32, 48, 64]), dropout=t.suggest_float("dropout", 0, .5),
               wd=t.suggest_float("wd", 1e-6, 1e-2, log=True))
    with mlflow.start_run(run_name=f"trial_{t.number}", nested=True):
        mlflow.log_params(t.params); return fit(cfg, 8, t)

if __name__ == "__main__":
    s = run_study("task2_classifier", objective, 25, "maximize")
    mlflow.set_experiment("task2_classifier")
    with mlflow.start_run(run_name="final"):
        fit(s.best_params, 40, save="checkpoints/t2_clf.pt")
    # test-set report (fixed manifest, all 4 classes)
    c = torch.load("checkpoints/t2_clf.pt"); m = Classifier(c["cfg"]["base"], c["cfg"]["dropout"]).to(dev)
    m.load_state_dict(c["state"]); m.eval()
    tm = json.load(open("data/test_manifest.json")); dl = DataLoader(ManifestDS(te, tm), batch_size=256)
    P, T = [], []
    with torch.no_grad():
        for x, _, l, _ in dl: P += m(x.to(dev)).argmax(1).cpu().tolist(); T += l.tolist()
    print(classification_report(T, P, target_names=CLASSES, digits=4))
    np.save("checkpoints/t2_cm.npy", confusion_matrix(T, P, normalize="true"))