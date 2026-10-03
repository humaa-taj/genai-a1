import torch, torch.nn as nn, torch.nn.functional as F

class AE(nn.Module):
    """Conv encoder -> linear latent bottleneck -> conv decoder. No skips (ablate one later if you want)."""
    def __init__(self, base=32, latent=256, dropout=0.1):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]; self.c3 = c[3]
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1, bias=False), nn.BatchNorm2d(o), nn.LeakyReLU(.2, True))
        def ublk(i, o): return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))
        self.enc = nn.Sequential(blk(3, c[0]), blk(c[0], c[1]), blk(c[1], c[2]), blk(c[2], c[3]))  # 128 -> 8
        self.fc_e = nn.Linear(c[3] * 64, latent); self.drop = nn.Dropout(dropout)
        self.fc_d = nn.Linear(latent, c[3] * 64)
        self.dec = nn.Sequential(ublk(c[3], c[2]), ublk(c[2], c[1]), ublk(c[1], c[0]),
                                 nn.ConvTranspose2d(c[0], 3, 4, 2, 1), nn.Sigmoid())
    def forward(self, x):
        z = self.drop(self.fc_e(self.enc(x).flatten(1)))
        return self.dec(self.fc_d(z).view(-1, self.c3, 8, 8))

class Classifier(nn.Module):
    def __init__(self, base=32, dropout=0.3, n=4):
        super().__init__()
        ch, L, i = [base, base * 2, base * 4, base * 8], [], 3
        for o in ch:
            L += [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                  nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True), nn.MaxPool2d(2)]
            i = o
        self.f = nn.Sequential(*L, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout))
        self.head = nn.Linear(i, n)
    def forward(self, x): return self.head(self.f(x))          # logits

class SoftMoE(nn.Module):
    """w = softmax(gate(x)/T); y = w0*x + w1*salt(x) + w2*blur(x) + w3*occ(x)."""
    def __init__(self, gate, experts, T=1.0):
        super().__init__(); self.gate, self.experts, self.T = gate, nn.ModuleList(experts), T
    def forward(self, x):
        logits = self.gate(x); w = F.softmax(logits / self.T, 1)
        outs = [x] + [e(x) for e in self.experts]
        y = sum(w[:, i, None, None, None] * o for i, o in enumerate(outs))
        return y, w, logits

# ---------------- Task 4 ----------------
class UNetG(nn.Module):
    def __init__(self, base=64, emb=16, dropout=0.3, n_styles=3):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 8, base * 8]   # 64,32,16,8,4,2
        self.emb = nn.Embedding(n_styles, emb)
        def down(i, o, norm=True):
            return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1, bias=False), *( [nn.BatchNorm2d(o)] if norm else []), nn.LeakyReLU(.2, True))
        def up(i, o, d): return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1, bias=False), nn.BatchNorm2d(o),
                                              *([nn.Dropout(dropout)] if d else []), nn.ReLU(True))
        self.d = nn.ModuleList([down(3 + emb, c[0], False), down(c[0], c[1]), down(c[1], c[2]),
                                down(c[2], c[3]), down(c[3], c[4]), down(c[4], c[5])])
        self.u = nn.ModuleList([up(c[5] + emb, c[4], True), up(c[4] * 2, c[3], True), up(c[3] * 2, c[2], True),
                                up(c[2] * 2, c[1], False), up(c[1] * 2, c[0], False)])
        self.final = nn.ConvTranspose2d(c[0] * 2, 3, 4, 2, 1)
    def forward(self, x, style):
        m = self.emb(style)[:, :, None, None]
        h = torch.cat([x, m.expand(-1, -1, x.shape[2], x.shape[3])], 1)
        sk = []
        for d in self.d: h = d(h); sk.append(h)
        h = self.u[0](torch.cat([h, m.expand(-1, -1, h.shape[2], h.shape[3])], 1))
        for k in range(1, 5): h = self.u[k](torch.cat([h, sk[-1 - k]], 1))
        return torch.tanh(self.final(torch.cat([h, sk[0]], 1)))

class PatchD(nn.Module):
    def __init__(self, base=64, emb=16, n_styles=3):
        super().__init__()
        self.emb = nn.Embedding(n_styles, emb)
        def blk(i, o, s): return nn.Sequential(nn.Conv2d(i, o, 4, s, 1, bias=False), nn.BatchNorm2d(o), nn.LeakyReLU(.2, True))
        self.net = nn.Sequential(nn.Conv2d(6 + emb, base, 4, 2, 1), nn.LeakyReLU(.2, True),
                                 blk(base, base * 2, 2), blk(base * 2, base * 4, 2), blk(base * 4, base * 8, 1),
                                 nn.Conv2d(base * 8, 1, 4, 1, 1))
    def forward(self, photo, sketch, style):
        m = self.emb(style)[:, :, None, None].expand(-1, -1, photo.shape[2], photo.shape[3])
        return self.net(torch.cat([photo, sketch, m], 1))