import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import torch
import torch.nn as nn
import math
from IPython.display import display, HTML

SEQ_LEN = 60

NUM_LAYERS  = 2
EMB_DIM     = 64
TIME_DIM    = 32
F           = 63          
HIDDEN      = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class TimeEmbedding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe)  # (max_len, d_model)
    def forward(self, T):
        return self.pe[:T]  # (T,d_model)

class CondLSTMGenerator(nn.Module):
    """
    Training (teacher-forcing): input = [prev_frame, label_emb, time_emb] -> predict current frame
    Inference (generate): feed zeros (or seed) as prev, roll out T steps for a label
    """
    def __init__(self, F=63, num_classes=10, emb_dim=64, time_dim=32, hidden=256, layers=2):
        super().__init__()
        self.F = F
        self.label_emb = nn.Embedding(num_classes, emb_dim)
        self.time_emb  = TimeEmbedding(time_dim, max_len=2048)
        self.in_lin    = nn.Linear(F + emb_dim + time_dim, hidden)
        self.lstm      = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True)
        self.out_lin   = nn.Linear(hidden, F)

    def forward_teacher_forcing(self, x, y_labels):
        """
        x: (B,T,F) ground truth standardized sequence
        y_labels: (B,) label indices
        returns preds: (B,T,F)
        """
        B,T,F = x.shape
        lab = self.label_emb(y_labels)           # (B,emb)
        lab = lab[:,None,:].expand(B,T,-1)       # (B,T,emb)
        te  = self.time_emb(T)[None,:,:].expand(B,-1,-1)  # (B,T,time)

        # teacher forcing: prev = shifted gt, first prev = zeros
        x_prev = torch.zeros_like(x)
        x_prev[:,1:,:] = x[:,:-1,:]

        h_in = torch.cat([x_prev, lab, te], dim=-1)  # (B,T,F+emb+time)
        h = self.in_lin(h_in)
        h,_ = self.lstm(h)
        pred = self.out_lin(h)                        # (B,T,F)
        return pred

    @torch.no_grad()
    def generate(self, y_labels, T, device, start_frame=None):
        """
        y_labels: (B,)
        T: int length
        start_frame: optional (B,F) seed; else zeros
        returns: (B,T,F)
        """
        B = y_labels.shape[0]
        lab = self.label_emb(y_labels).to(device)       # (B,emb)
        te  = self.time_emb(T).to(device)               # (T,time)
        state = None
        if start_frame is None:
            prev = torch.zeros(B, 1, self.F, device=device)
        else:
            prev = start_frame[:,None,:].to(device)

        outs = []
        for t in range(T):
            lab_t = lab[:,None,:]                      # (B,1,emb)
            te_t  = te[t].expand(B,1,-1)               # (B,1,time)
            h_in  = torch.cat([prev, lab_t, te_t], dim=-1)  # (B,1,F+emb+time)
            h0    = self.in_lin(h_in)
            h1, state = self.lstm(h0, state)
            out   = self.out_lin(h1)                   # (B,1,F)
            outs.append(out)
            prev = out                                 # autoregressive
        return torch.cat(outs, dim=1)                  # (B,T,F)

def masked_mse(pred, target, mask):
    # pred/target: (B,T,F), mask: (B,T)
    diff = (pred - target)**2
    diff = diff.mean(dim=-1)      # (B,T)
    diff = diff * mask
    denom = mask.sum(dim=1) + 1e-6
    return (diff.sum(dim=1)/denom).mean()

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

def unstandardize(seq_norm, mean, std):
    return seq_norm * std + mean
    

def _bounds_from_seq(seq_unstd):
    """Compute robust axis limits from all frames."""
    coords = seq_unstd.reshape(-1, 21, 3)
    xy_all = coords[:, :, :2].reshape(-1, 2)
    good = ~np.isnan(xy_all).any(axis=1) & ~np.isinf(xy_all).any(axis=1)
    xy_all = xy_all[good]
    if xy_all.size == 0:
        return -1, 1, -1, 1
    xmin, ymin = xy_all.min(axis=0)
    xmax, ymax = xy_all.max(axis=0)
    pad = 0.1 * max(1e-6, xmax - xmin, ymax - ymin)
    return xmin - pad, xmax + pad, ymin - pad, ymax + pad

def animate_sequence(seq_unstd, title="Hand animation", mask=None, fps=30, save_path=None):
    """
    seq_unstd: (T,63) UNstandardized
    mask: optional (T,) 1=valid 0=invalid; invalid frames are 'held'
    """
    T = seq_unstd.shape[0]
    coords = seq_unstd.reshape(T, 21, 3)[:, :, :2]
    if mask is not None and len(mask) == T:
        coords_filled = coords.copy()
        last = coords_filled[0]
        for t in range(T):
            if mask[t] > 0.5 and np.isfinite(coords_filled[t]).all():
                last = coords_filled[t]
            else:
                coords_filled[t] = last
        coords = coords_filled

    xmin, xmax, ymin, ymax = _bounds_from_seq(seq_unstd)
    fig, ax = plt.subplots()
    scat = ax.scatter([], [])
    lines = [ax.plot([], [])[0] for _ in HAND_CONNECTIONS]
    ax.set_xlim(xmin, xmax); ax.set_ylim(-ymax, -ymin); ax.set_aspect('equal', adjustable='box')

    def init():
        scat.set_offsets(np.empty((0,2)))
        for ln in lines: ln.set_data([], [])
        ax.set_title(title)
        return (scat, *lines)

    def update(t):
        xy = coords[t]
        scat.set_offsets(np.c_[xy[:,0], -xy[:,1]])
        for ln, (a,b) in zip(lines, HAND_CONNECTIONS):
            ln.set_data([xy[a,0], xy[b,0]], [-xy[a,1], -xy[b,1]])
        ax.set_title(f"{title}  t={t+1}/{T}")
        return (scat, *lines)

    anim = FuncAnimation(fig, update, init_func=init, frames=T, interval=1000/max(1,fps), blit=True)
    if save_path:
        try:
            anim.save(save_path, fps=fps, dpi=120)
            plt.close(fig)
            print(f"Saved animation to {save_path}")
        except Exception as e:
            print(f"Saving failed ({e}). Showing inline instead.")
            plt.show()
            #plt.close(fig); display(HTML(anim.to_jshtml()))
    else:
    	plt.show()
        #plt.close(fig); display(HTML(anim.to_jshtml()))

def generate_sign(sign_name, length=SEQ_LEN, seed_frame=None, fps=30, save_path=None):
    """Generate a sequence given the sign label, then animate."""
    if sign_name not in LABEL_MAP:
        raise ValueError(f"Unknown sign '{sign_name}'. Known: {list(LABEL_MAP.keys())}")
    y_lbl = torch.tensor([LABEL_MAP[sign_name]], device=DEVICE)
    model.eval()
    with torch.no_grad():
        if seed_frame is not None:
            seed = torch.from_numpy(seed_frame).float().to(DEVICE)
        else:
            seed = None
        gen = model.generate(y_labels=y_lbl, T=length, device=DEVICE, start_frame=seed).cpu().numpy()[0]
    gen_unstd = unstandardize(gen, TRAIN_MEAN, TRAIN_STD)
    animate_sequence(gen_unstd, title=f"Generated – {sign_name}", fps=fps, save_path=save_path)

def load_checkpoint(path="cond_gen.pt"):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = ckpt.get("config", {})
    mdl = CondLSTMGenerator(F=cfg.get("F", F),
                            num_classes=len(ckpt["label_map"]),
                            emb_dim=cfg.get("EMB_DIM", EMB_DIM),
                            time_dim=cfg.get("TIME_DIM", TIME_DIM),
                            hidden=cfg.get("HIDDEN", HIDDEN),
                            layers=cfg.get("NUM_LAYERS", NUM_LAYERS)).to(DEVICE)
    mdl.load_state_dict(ckpt["model_state"])
    mdl.eval()
    print("Loaded model.")
    return mdl, ckpt["mean"], ckpt["std"], ckpt["label_map"]

model, TRAIN_MEAN, TRAIN_STD, LABEL_MAP = load_checkpoint("cond_gen.pt")
generate_sign("home", length=SEQ_LEN)





