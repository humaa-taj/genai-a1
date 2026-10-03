import json, numpy as np, torch, torch.nn.functional as F, mlflow, optuna
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pytorch_msssim import ssim
from common.models import UNetG, PatchD
from common.train_utils import dev, run_study

R = "data/FS2K"
KEY_PHOTO, KEY_SKETCH, KEY_STYLE = "image_name", "sketch_name", "category"   # <- VERIFY against the json

def read(path):
    return np.asarray(Image.open(path).convert("RGB").resize((128, 128), Image.BICUBIC))

class FS2K(Dataset):
    def __init__(self, items, train=False):
        self.train = train
        self.P = np.stack([read(f"{R}/{a['photo']}") for a in items])
        self.S = np.stack([read(f"{R}/{a['sketch']}") for a in items])
        self.st = np.array([a["style"] for a in items])
    def __len__(self): return len(self.P)
    def __getitem__(self, i):
        p, s = self.P[i], self.S[i]
        if self.train and np.random.rand() < .5: p, s = p[:, ::-1], s[:, ::-1]      # SAME flip for both
        f = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float() / 127.5 - 1
        return f(p), f(s), int(self.st[i])

def load_items(split):
    A = json.load(open(f"{R}/anno_{split}.json"))
    items = [dict(photo=a[KEY_PHOTO], sketch=a[KEY_SKETCH], style=int(a[KEY_STYLE])) for a in A]
    lo = min(i["style"] for i in items)
    for i in items: i["style"] -= lo
    return items

train_all = load_items("train")
tr_idx, va_idx = train_test_split(range(len(train_all)), test_size=.15, random_state=42,
                                  stratify=[a["style"] for a in train_all])
TR = FS2K([train_all[i] for i in tr_idx], True); VA = FS2K([train_all[i] for i in va_idx])
fixed = [VA[i] for i in range(8)]       # same validation photos logged every time

@torch.no_grad()
def validate(G, vl):
    G.eval(); l1 = s = n = 0
    for p, y, st in vl:
        p, y, st = p.to(dev), y.to(dev), st.to(dev); g = G(p, st); b = len(p)
        l1 += F.l1_loss(g, y).item() * b; s += ssim((g + 1) / 2, (y + 1) / 2, data_range=1.0).item() * b; n += b
    return l1 / n, s / n

def log_samples(G, ep):
    G.eval()
    with torch.no_grad():
        p = torch.stack([f[0] for f in fixed]).to(dev); st = torch.tensor([f[2] for f in fixed]).to(dev)
        g = G(p, st); y = torch.stack([f[1] for f in fixed]).to(dev)
        grid = torch.cat([torch.cat(list(t), 2) for t in (p, g, y)], 1)      # photo / generated / target rows
        mlflow.log_image(((grid.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).astype("uint8"), f"samples/ep_{ep:03d}.png")

def train_gan(cfg, epochs, trial=None, save=None):
    G = UNetG(cfg["base"], cfg["emb"], cfg["dropout"]).to(dev); D = PatchD(cfg["base"], cfg["emb"]).to(dev)
    oG = torch.optim.Adam(G.parameters(), cfg["lr_g"], betas=(.5, .999))
    oD = torch.optim.Adam(D.parameters(), cfg["lr_d"], betas=(.5, .999))
    tl = DataLoader(TR, cfg["bs"], shuffle=True, drop_last=True, num_workers=2); vl = DataLoader(VA, 32)
    bce = F.binary_cross_entropy_with_logits; best = 1e9
    for ep in range(epochs):
        G.train(); D.train(); acc = dict(d_real=0, d_fake=0, g_adv=0, g_rec=0)
        for p, y, st in tl:
            p, y, st = p.to(dev), y.to(dev), st.to(dev); fake = G(p, st)
            rl, fl = D(p, y, st), D(p, fake.detach(), st)
            d_real, d_fake = bce(rl, torch.ones_like(rl)), bce(fl, torch.zeros_like(fl))
            oD.zero_grad(); (.5 * (d_real + d_fake)).backward(); oD.step()
            gl = D(p, fake, st); g_adv = bce(gl, torch.ones_like(gl)); g_rec = F.l1_loss(fake, y)
            oG.zero_grad(); (g_adv + cfg["lam"] * g_rec).backward(); oG.step()
            for k, v in zip(acc, (d_real, d_fake, g_adv, g_rec)): acc[k] += v.item() / len(tl)
        l1, s = validate(G, vl); obj = l1 + 1 - s
        mlflow.log_metrics({**acc, "val_l1": l1, "val_ssim": s, "val_obj": obj}, step=ep)
        if not trial and ep % 10 == 0: log_samples(G, ep)
        if obj < best:
            best = obj
            if save: torch.save(dict(cfg=cfg, state=G.state_dict()), save)
        if trial:
            trial.report(obj, ep)
            if trial.should_prune(): raise optuna.TrialPruned()
    return best

def objective(t):
    cfg = dict(lr_g=t.suggest_float("lr_g", 5e-5, 1e-3, log=True), lr_d=t.suggest_float("lr_d", 5e-5, 1e-3, log=True),
               bs=t.suggest_categorical("bs", [8, 16, 32]), base=t.suggest_categorical("base", [32, 48, 64]),
               dropout=t.suggest_float("dropout", 0, .5), emb=t.suggest_categorical("emb", [8, 16, 32, 64]),
               lam=t.suggest_float("lam", 10, 200, log=True))
    with mlflow.start_run(run_name=f"trial_{t.number}", nested=True):
        mlflow.log_params(t.params); return train_gan(cfg, 15, t)

if __name__ == "__main__":
    s = run_study("task4_cgan", objective, 20); mlflow.set_experiment("task4_cgan")
    with mlflow.start_run(run_name="final"):
        mlflow.log_params(s.best_params); train_gan(s.best_params, 200, save="checkpoints/t4_gen.pt")