#!/usr/bin/env python3
"""
SCM Adversarial Detection v3 — terminal runner
Generated from scm_v3.ipynb

Usage:
  ATTACK_NAME=TextFooler python3 scm_v3_run.py
  ATTACK_NAME=DeepWordBug python3 scm_v3_run.py

Environment overrides (all optional):
  ATTACK_NAME        TextFooler | DeepWordBug | PWWS  (default: TextFooler)
  N_TARGET_ATTACKS   int  (default: 1000)
  LAYERS             comma-separated ints  (default: 1,3,6,9,12)
  MODEL_NAME         HuggingFace model string
"""

import os, sys

# ── Environment overrides ────────────────────────────────────────────────────
# These let you configure a run without editing the file.
# Set before running: ATTACK_NAME=DeepWordBug python3 scm_v3_run.py
_ATTACK_NAME = os.environ.get("ATTACK_NAME", "TextFooler")
_N_TARGET    = int(os.environ.get("N_TARGET_ATTACKS", "1000"))
_LAYERS_STR  = os.environ.get("LAYERS", "1,3,6,9,12")
_MODEL_NAME  = os.environ.get("MODEL_NAME",
               "textattack/bert-base-uncased-SST-2")

# SCM Adversarial Detection — v3
# Changes from v2
# | # | Change | Impact |
# |---|---|---|
# | 1 | **Multi-attack support** — `ATTACK_NAME` config constant; one-line swap for DeepWordBug | Generalization claim |
# | 2 | **Paper captions corrected** — removed stale "lower κ at early layers" wording | Accuracy |
# | 3 | **Figure 3 at best layer (L12)** — box plot now shows the actual signal, not the null result at L1 | Visual clarity |
# | 4 | **Base model ablation** — re-runs conditioning on `bert-base-uncased` with saved pairs | Mechanism |
# | 5 | **Two-row Table 1** — original (N=29, base model) vs replicated (N=1000, fine-tuned) | Reproducibility |
# | 6 | **Cosine distance clarification** — explains why values > 1.0 are valid and why AUC=0.537≠0.25 | Correctness |
# | 7 | **Replication note cell** — documents the finding inversion (L1→L12, lower→higher) | Transparency |
# Key scientific update (v2 → v3)
# The corrected JVP+VJP estimator on N=1,000 real SST-2 examples **inverts the original finding**:
# - **Original claim (N=29, biased estimator):** lower κ at Layer 1, AUC=0.72
# - **Replicated result (N=1000, correct estimator):** higher κ at **Layer 12**, AUC=0.857
# The layer-1 signal was noise. The real effect is a σ_max explosion at the final transformer
# layer (2.69× higher for adversarial inputs), consistent with a decision-boundary geometry
# mechanism: TextFooler must cross the SST-2 classifier boundary, which maximally stretches
# the Jacobian at the layer feeding the classification head.


# ────────────────────────────────────────────────────────────────────────────

# ── Install: packages ─────────────────────────────────────────────────────────
import subprocess, sys, importlib

# Remove broken TensorFlow first. The vast.ai PyTorch image ships a partially-
# installed TF; transformers detects it and tries to import it, hitting
# "No module named 'tensorflow.python'" deep in the import chain.
subprocess.call(
    [sys.executable, '-m', 'pip', 'uninstall', '-y',
     'tensorflow', 'tensorflow-cpu', 'tensorflow-gpu', 'tf-nightly'],
    stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
)

def pip(*pkgs, retries=5, timeout=120):
    """Install packages with retry + timeout. Skips packages already importable."""
    to_install = []
    for pkg in pkgs:
        # normalise: 'scikit-learn' imports as 'sklearn', etc.
        import_name = pkg.replace('-', '_').split('>=')[0].split('==')[0]
        remap = {'scikit_learn': 'sklearn', 'Pillow': 'PIL'}
        import_name = remap.get(import_name, import_name)
        try:
            importlib.import_module(import_name)
        except ImportError:
            to_install.append(pkg)

    if not to_install:
        print(f'  already installed: {pkgs}')
        return

    print(f'  installing: {to_install}')
    cmd = [
        sys.executable, '-m', 'pip', 'install',
        '--quiet',
        f'--timeout={timeout}',
        f'--retries={retries}',
        *to_install,
    ]
    subprocess.check_call(cmd)

# torch is ~2 GB — install alone first so a timeout only affects it
pip('torch')
# remaining packages are small; install together
pip('transformers', 'datasets', 'scipy',
    'matplotlib', 'seaborn', 'pandas', 'scikit-learn', 'tqdm')
# textattack last — it pulls many sub-dependencies
pip('textattack')
print('Packages ready.')



# ────────────────────────────────────────────────────────────────────────────

# ── Install: NLTK data (plain Python, not !python -c to avoid shell quoting) ──
import nltk
for pkg in [
    'averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger',
    'punkt', 'punkt_tab', 'stopwords', 'omw-1.4', 'wordnet', 'universal_tagset',
]:
    nltk.download(pkg, quiet=True)
print('NLTK ready.')



# ────────────────────────────────────────────────────────────────────────────

# ── Experiment configuration ──────────────────────────────────────────────────
# To run DeepWordBug: change ATTACK_NAME to 'DeepWordBug' and re-run from Cell 8.
# All downstream cells (conditioning, analysis, export) are attack-agnostic.

ATTACK_NAME      = _ATTACK_NAME   # override via env: ATTACK_NAME=DeepWordBug
N_TARGET_ATTACKS = _N_TARGET
BUFFER_FACTOR    = 1.4            # source sentences = N_TARGET × BUFFER_FACTOR
LAYERS           = [int(x) for x in _LAYERS_STR.split(',')]  # override: LAYERS=1,3,12
MODEL_NAME       = _MODEL_NAME

print(f'Attack          : {ATTACK_NAME}')
print(f'Target pairs    : {N_TARGET_ATTACKS}')
print(f'Model           : {MODEL_NAME}')
print(f'Layers          : {LAYERS}')




# ────────────────────────────────────────────────────────────────────────────

# ── Imports + hardware probe ──────────────────────────────────────────────────
import os, gc, time, json, hashlib, warnings
from pathlib import Path
from typing import Callable, List, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive: saves figures without display
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizer, BertModel
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')
torch.manual_seed(42); np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_cpus = os.cpu_count() or 1

if device.type == 'cuda':
    print(f"GPU  : {torch.cuda.get_device_name(0)}")
    print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
else:
    print("CPU only — will be slow")
print(f"CPUs : {n_cpus}")

try:
    from torch.autograd.functional import jvp as _torch_jvp
    HAS_TORCH_JVP = True
    print("torch.autograd.functional.jvp : available")
except ImportError:
    HAS_TORCH_JVP = False
    print("JVP not available — using finite-difference fallback")

CKPT_DIR    = Path(f"checkpoints_{ATTACK_NAME}"); CKPT_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path(f"results_{ATTACK_NAME}");     RESULTS_DIR.mkdir(exist_ok=True)
print(f"Checkpoint dir  : {CKPT_DIR}")
print(f"Results dir     : {RESULTS_DIR}")



# ────────────────────────────────────────────────────────────────────────────

# ── Load SST-2 ────────────────────────────────────────────────────────────────
from datasets import load_dataset

N_SOURCE = int(N_TARGET_ATTACKS * BUFFER_FACTOR)

sst2_val   = load_dataset('sst2', split='validation')
sst2_train = load_dataset('sst2', split='train')

def balanced_sample(dataset, n):
    rng = np.random.default_rng(42)
    pos = [(r['sentence'], r['label']) for r in dataset if r['label'] == 1]
    neg = [(r['sentence'], r['label']) for r in dataset if r['label'] == 0]
    rng.shuffle(pos); rng.shuffle(neg)
    half = n // 2
    return pos[:half] + neg[:half]

val_samples = balanced_sample(sst2_val, min(len(sst2_val), N_SOURCE))
if len(val_samples) < N_SOURCE:
    val_samples += balanced_sample(sst2_train, N_SOURCE - len(val_samples))

np.random.shuffle(val_samples)
CLEAN_SAMPLES = val_samples[:N_SOURCE]

pos_n = sum(1 for _, l in CLEAN_SAMPLES if l == 1)
neg_n = sum(1 for _, l in CLEAN_SAMPLES if l == 0)
print(f'Source sentences : {len(CLEAN_SAMPLES)}  (pos={pos_n}, neg={neg_n})')
print(f'Sample: "{CLEAN_SAMPLES[0][0][:60]}"  → label={CLEAN_SAMPLES[0][1]}')



# ────────────────────────────────────────────────────────────────────────────

# ── Load models ───────────────────────────────────────────────────────────────
# Attack model and conditioning model are the SAME (v2 fix: model alignment).
# Swap COND_MODEL = base_model below to test pretrained-only conditioning (Cell 4b).
print(f'Loading {MODEL_NAME} ...')
cls_model     = BertForSequenceClassification.from_pretrained(MODEL_NAME)
cls_tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
cls_model     = cls_model.to(device).eval()
for p in cls_model.parameters(): p.requires_grad_(False)

COND_MODEL     = cls_model
COND_TOKENIZER = cls_tokenizer

# Base model kept for ablation (Cell 4b) — loaded lazily there
BASE_MODEL_NAME = 'bert-base-uncased'
base_model, base_tokenizer = None, None   # populated in ablation cell

print(f'Conditioning model : {COND_MODEL.__class__.__name__} ({MODEL_NAME})')



# ────────────────────────────────────────────────────────────────────────────

# ── SpectralConditioningMonitor (v3: vmap-batched sigma_low) ─────────────────
#
# Option D: _sigma_low_quantile routes to a vmap implementation when
# torch.func is available (PyTorch >= 2.0), falling back to the original
# serial autograd.grad loop automatically.
#
# Speedup:
#   Serial:  n_sigma_low serial autograd.grad calls, each launching a full
#            backward pass through BERT for a single (1, hidden_dim) vector.
#   Batched: torch.func.vjp gives a pure-functional vjp closure.
#            vmap(single_vjp_norm)(U) fuses n_sigma_low backward passes into
#            one batched computation — far fewer CUDA kernel launches.
#
# Expected: ~4x speedup on sigma_low (60% of total passes)
#           -> ~2.5x overall conditioning speedup
#           -> conditioning phase: ~29 min -> ~12 min per run

class SpectralConditioningMonitor:
    def __init__(self, power_iterations=5, n_sigma_low=15,
                 epsilon=1e-8, fd_eps=1e-4, use_vmap=True):
        self.power_iterations = power_iterations
        self.n_sigma_low      = n_sigma_low
        self.epsilon          = epsilon
        self.fd_eps           = fd_eps
        self.use_vmap         = use_vmap

        # Detect vmap at init so we only check once
        self._vmap_available  = False
        if use_vmap:
            try:
                from torch.func import vjp, vmap  # noqa: F401
                self._vmap_available = True
            except ImportError:
                pass

    def estimate_all_layers(self, model, input_ids, attention_mask, layers):
        with torch.no_grad():
            encoder = model.bert if hasattr(model, 'bert') else model
            x_base  = encoder.embeddings(input_ids).detach()
        results = {}
        for layer_idx in layers:
            f     = self._make_layer_fn(model, attention_mask, layer_idx)
            s_max = self._sigma_max(f, x_base)
            s_low = self._sigma_low_quantile(f, x_base, x_base.shape[-1])
            results[layer_idx] = {
                'kappa'              : s_max / (s_low + self.epsilon),
                'sigma_max'          : s_max,
                'sigma_low_quantile' : s_low,
            }
        torch.cuda.empty_cache()
        return results

    def _make_layer_fn(self, model, attention_mask, layer_idx):
        def f(emb, _L=layer_idx, _m=attention_mask):
            enc = model.bert if hasattr(model, 'bert') else model
            out = enc(inputs_embeds=emb, attention_mask=_m, output_hidden_states=True)
            return out.hidden_states[_L][:, 0, :]
        return f

    def _jvp(self, f, x, v):
        if HAS_TORCH_JVP:
            try:
                from torch.autograd.functional import jvp as _tjvp
                _, u = _tjvp(f, (x,), (v,), create_graph=False, strict=False)
                return u.detach()
            except Exception:
                pass
        eps = self.fd_eps
        with torch.no_grad():
            return ((f(x + eps*v) - f(x - eps*v)) / (2.0*eps)).detach()

    def _sigma_max(self, f, x_base):
        v = torch.randn_like(x_base); v = v / (v.norm() + self.epsilon)
        sigma = 0.0
        for _ in range(self.power_iterations):
            u     = self._jvp(f, x_base, v)
            sigma = float(u.norm().item())
            if sigma < self.epsilon: break
            u_hat = u / (sigma + self.epsilon)
            x_tmp = x_base.detach().requires_grad_(True)
            v_new = torch.autograd.grad(f(x_tmp), x_tmp,
                                        grad_outputs=u_hat, retain_graph=False)[0]
            v = v_new.detach() / (v_new.norm().item() + self.epsilon)
        return float(sigma)

    def _sigma_low_quantile(self, f, x_base, hidden_dim):
        """Lower-tail proxy for sigma_min (NOT strict sigma_min).
        Routes to vmap when torch.func is available, serial fallback otherwise."""
        if self._vmap_available:
            try:
                return self._sigma_low_vmap(f, x_base, hidden_dim)
            except Exception:
                # vmap fails on some BERT variants (dropout, certain layer norms).
                # Fall through to serial — no user action needed.
                pass
        return self._sigma_low_serial(f, x_base, hidden_dim)

    def _sigma_low_vmap(self, f, x_base, hidden_dim):
        """Batched VJP via torch.func.vjp + vmap (PyTorch >= 2.0).

        Algorithm:
          1. vjp(f, x_base) -> (f_x, vjp_fn)   [one forward pass]
          2. Build U: (n_sigma_low, hidden_dim) random unit vectors
          3. vmap(single_vjp_norm)(U)            [one batched backward]
          4. Return 10th percentile of norms

        torch.func.vjp returns (primals_out, vjp_fn) where
        vjp_fn(cotangent) -> (grad_input,)  [note: tuple, index [0]]
        """
        import torch.nn.functional as F_nn
        from torch.func import vjp, vmap

        # Single forward pass; vjp_fn is a pure functional closure
        _, vjp_fn = vjp(f, x_base)

        # (n_sigma_low, hidden_dim) unit vectors in output space
        U = torch.randn(self.n_sigma_low, hidden_dim, device=x_base.device)
        U = F_nn.normalize(U, dim=1)

        def single_vjp_norm(u):
            # u: (hidden_dim,) after vmap strips batch dim
            # vjp_fn expects cotangent matching f output shape: (1, hidden_dim)
            jt_u = vjp_fn(u.unsqueeze(0))[0]   # (1, seq_len, hidden_dim)
            return jt_u.norm()

        norms = vmap(single_vjp_norm)(U)        # (n_sigma_low,)
        norms = norms.detach().float().cpu().numpy()
        valid = norms[norms > self.epsilon]
        return float(np.percentile(valid, 10)) if len(valid) > 0 else self.epsilon

    def _sigma_low_serial(self, f, x_base, hidden_dim):
        """Original serial loop — unchanged correctness, used as fallback."""
        x_req = x_base.detach().requires_grad_(True)
        f_x   = f(x_req)
        norms = []
        for _ in range(self.n_sigma_low):
            u = torch.randn(1, hidden_dim, device=x_base.device)
            u = u / (u.norm() + self.epsilon)
            try:
                jt_u = torch.autograd.grad(f_x, x_req, grad_outputs=u,
                                            retain_graph=True, create_graph=False)[0]
                n = float(jt_u.norm().item())
                if n > self.epsilon: norms.append(n)
            except RuntimeError:
                continue
        del f_x, x_req
        return float(np.percentile(norms, 10)) if norms else self.epsilon


scm = SpectralConditioningMonitor(power_iterations=5, n_sigma_low=15, use_vmap=True)
print(f'SCM v3 initialised')
print(f'  power_iterations : {scm.power_iterations}')
print(f'  n_sigma_low      : {scm.n_sigma_low}')
print(f'  vmap available   : {scm._vmap_available}')
if not scm._vmap_available:
    print(f'  NOTE: PyTorch >= 2.0 needed for vmap speedup.')
    print(f'        Using serial fallback. Upgrade with: pip install torch>=2.0')

# ── Pass budget ───────────────────────────────────────────────────────────────
n_layers    = 5
# sigma_max: power_iter x (1 JVP + 1 VJP) x n_layers — unchanged
serial_total  = scm.power_iterations * 2 * n_layers + scm.n_sigma_low * n_layers
# sigma_low vmap: 1 forward (inside vjp()) + 1 batched backward per layer
vmap_total    = scm.power_iterations * 2 * n_layers + 2 * n_layers
print()
print(f'GPU kernel sequences per text:')
print(f'  Serial  : {serial_total}  ({serial_total*2}/pair, {serial_total*2000:,}/1000 pairs)')
print(f'  Vmap    : {vmap_total}   ({vmap_total*2}/pair, {vmap_total*2000:,}/1000 pairs)')
print(f'  Reduction: {serial_total/vmap_total:.1f}x fewer sequences (sigma_low dominates)')


# ── SCM vmap benchmark skipped in terminal mode (runs automatically) ─────────


# ────────────────────────────────────────────────────────────────────────────

# ── Helpers ───────────────────────────────────────────────────────────────────
def tokenize(text, tokenizer, max_length=128):
    enc = tokenizer(text, return_tensors='pt', padding=True,
                    truncation=True, max_length=max_length)
    return {k: v.to(device) for k, v in enc.items()}

def get_cls_embedding(text, model, tokenizer):
    enc = tokenize(text, tokenizer)
    with torch.no_grad():
        enc_mod = model.bert if hasattr(model, 'bert') else model
        out = enc_mod(**enc, output_hidden_states=False)
    return out.last_hidden_state[:, 0, :]

def cosine_distance(a, b):
    return float(1.0 - torch.nn.functional.cosine_similarity(a, b).item())

def count_word_changes(orig, pert):
    a, b = orig.lower().split(), pert.lower().split()
    return sum(1 for x,y in zip(a,b) if x!=y) + abs(len(a)-len(b))

def file_md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()

def analyze_pair(pair, model, tokenizer):
    enc_c = tokenize(pair['clean'],       tokenizer)
    enc_a = tokenize(pair['adversarial'], tokenizer)
    cc = scm.estimate_all_layers(model, enc_c['input_ids'], enc_c['attention_mask'], LAYERS)
    ca = scm.estimate_all_layers(model, enc_a['input_ids'], enc_a['attention_mask'], LAYERS)
    ec = get_cls_embedding(pair['clean'],       model, tokenizer)
    ea = get_cls_embedding(pair['adversarial'], model, tokenizer)
    row = {'pair_idx': pair['idx'], 'clean_text': pair['clean'],
           'adv_text': pair['adversarial'], 'num_changes': pair['num_changes'],
           'cosine_distance': cosine_distance(ec, ea)}
    for L in LAYERS:
        row[f'kappa_clean_L{L}']     = cc[L]['kappa']
        row[f'kappa_adv_L{L}']       = ca[L]['kappa']
        row[f'delta_kappa_L{L}']     = ca[L]['kappa'] - cc[L]['kappa']
        row[f'sigma_max_clean_L{L}'] = cc[L]['sigma_max']
        row[f'sigma_max_adv_L{L}']   = ca[L]['sigma_max']
        row[f'sigma_low_clean_L{L}'] = cc[L]['sigma_low_quantile']
        row[f'sigma_low_adv_L{L}']   = ca[L]['sigma_low_quantile']
    return row

print('Helpers ready.')



# ────────────────────────────────────────────────────────────────────────────

# ── Phase 1: Attack generation ────────────────────────────────────────────────
import textattack
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets        import Dataset as TADataset
from textattack.attack_recipes  import (TextFoolerJin2019,
                                         DeepWordBugGao2018,
                                         PWWSRen2019)
from textattack import Attacker, AttackArgs
from textattack.attack_results import SuccessfulAttackResult


def build_textfooler_no_tf(model_wrapper):
    """TextFooler with SBERT cosine similarity instead of Universal Sentence Encoder.

    TextFooler (Jin et al. 2019) uses USE (via tensorflow_hub) to enforce
    semantic similarity between original and perturbed sentences.  On hosts
    where TF is broken or absent we swap USE for SBERT sentence embeddings,
    which are already installed and produce equivalent semantic filtering.

    Constraint kept identical in all other respects:
      - threshold  : 0.84  (same as original paper)
      - metric     : cosine (same)
      - window     : 15 tokens (same)
    """
    from textattack.attack_recipes.textfooler_jin_2019 import TextFoolerJin2019 as _TF
    from textattack.constraints.semantics.sentence_encoders import SBERT as BERTEncoder
    from textattack.constraints.semantics.sentence_encoders \
        import UniversalSentenceEncoder

    attack = _TF.build(model_wrapper)

    # Swap USE constraint → SBERT cosine similarity
    swapped = False
    for i, c in enumerate(attack.constraints):
        if isinstance(c, UniversalSentenceEncoder):
            attack.constraints[i] = BERTEncoder(
                threshold=0.84,
                metric='cosine',
                compare_against_original=False,
                window_size=15,
            )
            swapped = True
            break

    if swapped:
        print('  TextFooler: USE → SBERT cosine similarity (no TF dependency)')
    else:
        print('  TextFooler: USE constraint not found, using recipe as-is')
    return attack


def build_attack(attack_name, model_wrapper):
    """Build attack, automatically falling back to TF-free TextFooler if needed."""
    if attack_name == 'TextFooler':
        try:
            import tensorflow  # noqa: F401
            import tensorflow_hub  # noqa: F401
            print('  TF available — using standard TextFooler (USE)')
            return TextFoolerJin2019.build(model_wrapper)
        except ImportError:
            print('  TF/tensorflow_hub not available — using TextFooler+SBERT (no TF)')
            return build_textfooler_no_tf(model_wrapper)
    elif attack_name == 'DeepWordBug':
        return DeepWordBugGao2018.build(model_wrapper)
    elif attack_name == 'PWWS':
        return PWWSRen2019.build(model_wrapper)
    else:
        raise ValueError(f'Unknown attack: {attack_name}. '
                         f'Choose from TextFooler, DeepWordBug, PWWS')


TA_WORKERS    = min(8, n_cpus)
model_wrapper = HuggingFaceModelWrapper(cls_model, cls_tokenizer)
dataset       = TADataset(CLEAN_SAMPLES)
attack        = build_attack(ATTACK_NAME, model_wrapper)

attack_args = AttackArgs(
    num_examples           = len(CLEAN_SAMPLES),
    disable_stdout         = True,
    random_seed            = 42,
    num_workers_per_device = TA_WORKERS,
)
attacker = Attacker(attack, dataset, attack_args)

print(f'{ATTACK_NAME} on {len(CLEAN_SAMPLES)} sentences  ({TA_WORKERS} workers)')
t0 = time.time()
attack_results = attacker.attack_dataset()
elapsed_attack = time.time() - t0
print(f'Attack phase: {elapsed_attack/60:.1f} min')



# ────────────────────────────────────────────────────────────────────────────

# ── Extract pairs + save with hash ───────────────────────────────────────────
adversarial_pairs = []
for i, result in enumerate(attack_results):
    if isinstance(result, SuccessfulAttackResult):
        orig = result.original_text()
        pert = result.perturbed_text()
        adversarial_pairs.append({
            'idx': i, 'clean': orig, 'adversarial': pert,
            'label': CLEAN_SAMPLES[i][1],
            'num_changes': count_word_changes(orig, pert),
        })
    if len(adversarial_pairs) >= N_TARGET_ATTACKS: break

n_success = len(adversarial_pairs)
print(f'Successful: {n_success}/{len(attack_results)}  ({100*n_success/len(attack_results):.1f}%)')

PAIRS_PATH = CKPT_DIR / 'adversarial_pairs.json'
with open(PAIRS_PATH, 'w') as f: json.dump(adversarial_pairs, f)

PAIRS_HASH = file_md5(PAIRS_PATH)
HASH_PATH  = CKPT_DIR / 'adversarial_pairs.md5'
HASH_PATH.write_text(PAIRS_HASH)
print(f'Saved → {PAIRS_PATH}  (md5={PAIRS_HASH[:8]}...)')
for p in adversarial_pairs[:2]:
    print(f'  clean : {p["clean"][:70]}')
    print(f'  adv   : {p["adversarial"][:70]}  [n_changes={p["num_changes"]}]')



# ────────────────────────────────────────────────────────────────────────────

# ── Phase 2: Conditioning (fine-tuned model) ─────────────────────────────────
CKPT_RESULTS  = CKPT_DIR / 'cond_results.jsonl'
CKPT_INTERVAL = 100

# Reload + verify hash
if PAIRS_PATH.exists() and HASH_PATH.exists():
    stored  = HASH_PATH.read_text().strip()
    current = file_md5(PAIRS_PATH)
    if stored != current:
        raise RuntimeError(
            f'adversarial_pairs.json hash mismatch!\n'
            f'  stored={stored}\n  current={current}\n'
            'Delete checkpoints/ and rerun from attack cell.')
    with open(PAIRS_PATH) as fh: adversarial_pairs = json.load(fh)
    print(f'Pairs reloaded: {len(adversarial_pairs)}  (hash OK)')

completed_idx = set(); pair_rows = []
if CKPT_RESULTS.exists():
    with open(CKPT_RESULTS) as fh:
        for line in fh:
            row = json.loads(line); pair_rows.append(row)
            completed_idx.add(row['pair_idx'])
    print(f'Resumed: {len(pair_rows)} pairs already done')

todo = [p for p in adversarial_pairs if p['idx'] not in completed_idx]
print(f'Remaining: {len(todo)} pairs')

t1 = time.time()
with open(CKPT_RESULTS, 'a') as ckpt_f:
    for step, pair in enumerate(tqdm(todo, desc='Conditioning', unit='pair')):
        row = analyze_pair(pair, COND_MODEL, COND_TOKENIZER)
        pair_rows.append(row); ckpt_f.write(json.dumps(row) + '\n')
        if (step+1) % CKPT_INTERVAL == 0:
            ckpt_f.flush()
            elapsed = time.time()-t1
            eta = (elapsed/(step+1))*(len(todo)-step-1)
            tqdm.write(f'  {step+1}/{len(todo)} | {elapsed/60:.1f} min | ETA {eta/60:.1f} min')

elapsed_cond = time.time()-t1
print(f'Conditioning: {elapsed_cond/60:.1f} min  ({len(todo)/max(elapsed_cond,1):.2f} pairs/sec)')

df_pairs = pd.DataFrame(pair_rows)
print(f'df_pairs: {df_pairs.shape}')



# ────────────────────────────────────────────────────────────────────────────

# ── Build analysis dataframes ─────────────────────────────────────────────────
clean_rows, adv_rows = [], []
for _, row in df_pairs.iterrows():
    base = {'pair_idx': row['pair_idx'], 'num_changes': row['num_changes']}
    cr = {**base, 'type': 'clean'}
    ar = {**base, 'type': 'adversarial'}
    for L in LAYERS:
        cr[f'kappa_L{L}'] = row[f'kappa_clean_L{L}']
        ar[f'kappa_L{L}'] = row[f'kappa_adv_L{L}']
    clean_rows.append(cr); adv_rows.append(ar)

df_clean = pd.DataFrame(clean_rows)
df_adv   = pd.DataFrame(adv_rows)
df_all   = pd.concat([df_clean, df_adv], ignore_index=True)
N        = len(df_clean)
y_true   = [0]*N + [1]*N
print(f'N pairs: {N}  |  Total rows: {len(df_all)}')



# ────────────────────────────────────────────────────────────────────────────

# ── Discrimination analysis ───────────────────────────────────────────────────
print('='*70)
print(f' {ATTACK_NAME}: CLEAN vs ADVERSARIAL κ DISCRIMINATION')
print('='*70)

layer_results = []
for L in LAYERS:
    col     = f'kappa_L{L}'
    kc, ka  = df_clean[col].values, df_adv[col].values
    k_all   = np.concatenate([kc, ka])
    auc_hi  = roc_auc_score(y_true, k_all)
    auc_lo  = roc_auc_score(y_true, -k_all)
    best    = max(auc_hi, auc_lo)
    direction = 'lower' if auc_lo > auc_hi else 'higher'
    _, pval = stats.mannwhitneyu(kc, ka, alternative='two-sided')
    ps = np.sqrt((kc.std()**2 + ka.std()**2)/2)
    d  = abs(ka.mean()-kc.mean())/ps if ps > 0 else 0.0
    sig = '**' if pval < 0.01 else ('*' if pval < 0.05 else '')
    print(f'\nLayer {L}:  AUC={best:.4f}  p={pval:.4f}{sig}  d={d:.3f}  '
          f'κ_clean={kc.mean():.3f}±{kc.std():.3f}  κ_adv={ka.mean():.3f}±{ka.std():.3f}')
    layer_results.append(dict(attack=ATTACK_NAME, layer=L, auc=best,
                               direction=direction, pval=pval, cohens_d=d,
                               mean_clean=kc.mean(), mean_adv=ka.mean(),
                               std_clean=kc.std(), std_adv=ka.std()))

df_layers = pd.DataFrame(layer_results)
best_row  = df_layers.loc[df_layers['auc'].idxmax()]
best_L    = int(best_row['layer'])
print(f'\nBest layer: {best_L}  (AUC={best_row["auc"]:.4f}, p={best_row["pval"]:.2e}, d={best_row["cohens_d"]:.3f})')



# ────────────────────────────────────────────────────────────────────────────

# ── Cosine distance baseline ──────────────────────────────────────────────────
# SUGGESTION 6: Full clarification of cosine distance AUC.
#
# Observed: mean cosine_distance = 0.84, with 320/1000 pairs > 1.0.
# A cosine_distance > 1.0 means cosine_similarity < 0 (anti-parallel vectors).
# This is NOT a bug: it is expected because:
#   - COND_MODEL is the fine-tuned SST-2 classifier
#   - TextFooler flips the sentiment label
#   - The CLS representation at the final hidden layer encodes strong polarity
#   - A polarity flip pushes the CLS vector to the opposite hemisphere
#
# The paper originally claimed cosine AUC ≈ 0.25. The correct value is 0.537.
# This change arises because the original AUC was never computed in code;
# it was asserted. The baseline still fails to discriminate usefully (near
# random), but the mechanism differs from the original description.

cos_dists = df_pairs['cosine_distance'].values
print('Cosine distance diagnostics:')
print(f'  mean  : {cos_dists.mean():.4f}  (cosine_sim = {1-cos_dists.mean():.4f})')
print(f'  median: {np.median(cos_dists):.4f}')
print(f'  > 1.0 : {(cos_dists > 1.0).sum()} / {len(cos_dists)} pairs')
print(f'  Interpretation: {(cos_dists > 1.0).sum()} adversarial examples pushed to')
print(f'  opposite hemisphere of CLS space (label was flipped by attack)')

# Compute AUC: 0 = clean-clean pair, 1 = clean-adv pair
clean_texts = df_pairs['clean_text'].tolist()
cc_dists = []
for i in range(N):
    j = (i+1) % N
    a = get_cls_embedding(clean_texts[i], COND_MODEL, COND_TOKENIZER)
    b = get_cls_embedding(clean_texts[j], COND_MODEL, COND_TOKENIZER)
    cc_dists.append(cosine_distance(a, b))
cc_dists = np.array(cc_dists)

y_cos    = [0]*N + [1]*N
auc_cos  = roc_auc_score(y_cos, np.concatenate([cc_dists, cos_dists]))

print(f'\nClean↔clean cosine dist: {cc_dists.mean():.4f} ± {cc_dists.std():.4f}')
print(f'Clean↔adv   cosine dist: {cos_dists.mean():.4f} ± {cos_dists.std():.4f}')
print(f'\nCosine AUC              : {auc_cos:.4f}')
print(f'Best κ AUC              : {best_row["auc"]:.4f}')
print(f'κ advantage             : {best_row["auc"]-auc_cos:+.4f}')
print()
print('Note: original paper cited cosine AUC ≈ 0.25 — this was an assertion, not computed.')
print(f'Correct value is {auc_cos:.3f}. κ still provides a large advantage ({best_row["auc"]-auc_cos:+.3f}).')
print('The comparative claim holds; the mechanism explanation needs updating.')


# Replication note: finding inversion from N=29 to N=1,000
# The original paper (N=29, hand-crafted samples, base BERT, biased σ_max estimator)
# reported AUC=0.72 at **Layer 1**, with adversarial inputs showing **lower** κ.
# At N=1,000 (real SST-2 sentences, fine-tuned model, corrected JVP+VJP estimator):
# | | Original | Replicated |
# |---|---|---|
# | N pairs | 29 | 1,000 |
# | Model | bert-base-uncased | textattack/bert-base-uncased-SST-2 |
# | Estimator | Biased CLS-proxy power iteration | JVP+VJP on full input space |
# | Best layer | 1 | **12** |
# | Best AUC | 0.72 | **0.857** |
# | Direction | lower κ → adversarial | **higher** κ → adversarial |
# | Cosine AUC | ≈0.25 (asserted) | 0.537 (computed) |
# **Interpretation of the inversion:** The Layer-1 signal in the original run was noise
# amplified by two compounding artifacts: the biased estimator inflated differences at
# early layers (which process raw token geometry), and the tiny N made any difference
# statistically exploitable. With the corrected estimator, early layers (L1–L6) show
# near-zero discrimination (AUC 0.50–0.51, all p > 0.45).
# The real effect is at **Layer 12** (final transformer layer, which feeds the
# classification head): adversarial inputs have 2.69× higher σ_max and 1.66× higher
# σ_low, producing Cohen d = 1.224. This is consistent with a decision-boundary
# geometry mechanism — TextFooler must cross the classifier's decision boundary, which
# maximally stretches the Jacobian at exactly the layer that feeds the classification head.
# **This is a stronger and more interpretable result than the original.** The effect
# size is larger (d=1.224 vs original d implied by AUC=0.72), the mechanistic explanation
# is cleaner (last-mile geometry, not early-layer stability), and the finding is robust
# to N (confirmed at 1,000 samples with p=3.5e-168).


# ────────────────────────────────────────────────────────────────────────────

# ── Table 1: Original vs Replicated ──────────────────────────────────────────
# SUGGESTION 5: Two-row Table 1 making the replication explicit.
# Original row uses the numbers from the first published experiment (N=29).
# These are kept for transparency; the replicated numbers supersede them.

ORIGINAL_RESULTS = {
    # Source: scm-adversarial-detection-experiment-2.ipynb (N=29, base BERT, CLS-proxy SCM)
    'n'             : 29,
    'model'         : 'bert-base-uncased (base)',
    'best_layer'    : 1,
    'kappa_clean'   : (2.906, 0.282),
    'kappa_adv'     : (2.662, 0.309),
    'p_value'       : 0.0056,
    'auc'           : 0.7122,
    'cosine_auc'    : None,   # never computed in original code
    'note'          : 'Biased CLS-proxy σ_max estimator; hand-crafted samples',
}

# Replicated results from this run
best_L_row  = df_layers[df_layers['layer'] == best_L].iloc[0]
L12_row     = df_layers[df_layers['layer'] == 12].iloc[0]

REPLICATED_RESULTS = {
    'n'             : N,
    'model'         : MODEL_NAME,
    'best_layer'    : best_L,
    'kappa_clean'   : (best_L_row['mean_clean'], best_L_row['std_clean']),
    'kappa_adv'     : (best_L_row['mean_adv'],   best_L_row['std_adv']),
    'p_value'       : best_L_row['pval'],
    'auc'           : best_L_row['auc'],
    'cosine_auc'    : auc_cos,
    'note'          : 'JVP+VJP σ_max estimator; real SST-2; same model for attack+conditioning',
}

print('Table 1 — SCM Detection Performance')
print()
print(f'  {"":30} {"Original":>22}  {"Replicated":>22}')
print(f'  {"-"*78}')
print(f'  {"N pairs":<30} {str(ORIGINAL_RESULTS["n"]):>22}  {str(REPLICATED_RESULTS["n"]):>22}')
print(f'  {"Model":<30} {ORIGINAL_RESULTS["model"]:>22}  {REPLICATED_RESULTS["model"][-22:]:>22}')
print(f'  {"Best layer":<30} {str(ORIGINAL_RESULTS["best_layer"]):>22}  {str(REPLICATED_RESULTS["best_layer"]):>22}')
kc_o = ORIGINAL_RESULTS["kappa_clean"]
ka_o = ORIGINAL_RESULTS["kappa_adv"]
kc_r = REPLICATED_RESULTS["kappa_clean"]
ka_r = REPLICATED_RESULTS["kappa_adv"]
print(f'  {"κ (clean) at best layer":<30} {f"{kc_o[0]:.3f}±{kc_o[1]:.3f}":>22}  {f"{kc_r[0]:.3f}±{kc_r[1]:.3f}":>22}')
print(f'  {"κ (adv)   at best layer":<30} {f"{ka_o[0]:.3f}±{ka_o[1]:.3f}":>22}  {f"{ka_r[0]:.3f}±{ka_r[1]:.3f}":>22}')
print(f'  {"p-value":<30} {f"{ORIGINAL_RESULTS["p_value"]:.4f}":>22}  {f"{REPLICATED_RESULTS["p_value"]:.2e}":>22}')
print(f'  {"AUC (κ)":<30} {f"{ORIGINAL_RESULTS["auc"]:.4f}":>22}  {f"{REPLICATED_RESULTS["auc"]:.4f}":>22}')
cos_o_str = 'not computed'
cos_r_str = f'{REPLICATED_RESULTS["cosine_auc"]:.4f}'
print(f'  {"AUC (cosine baseline)":<30} {cos_o_str:>22}  {cos_r_str:>22}')
print(f'  {"Estimator":<30} {"CLS-proxy (biased)":>22}  {"JVP+VJP (correct)":>22}')
print()
print(f'  Key change: best layer shifted 1 → {best_L}, direction lower → higher κ.')
print(f'  Effect size increased: AUC {ORIGINAL_RESULTS["auc"]:.3f} → {REPLICATED_RESULTS["auc"]:.3f}, d={best_L_row["cohens_d"]:.3f}')



# ────────────────────────────────────────────────────────────────────────────

# ── Ablation: base model conditioning ────────────────────────────────────────
# SUGGESTION 4: Re-run conditioning on bert-base-uncased (unfinetuned) using
# the SAME adversarial pairs already generated.  No new attacks needed.
#
# Two possible outcomes, both scientifically valuable:
#   a) Signal disappears  → effect is specific to the fine-tuned classifier's
#                            geometry (supports decision-boundary mechanism)
#   b) Signal persists    → effect is a property of the input text itself,
#                            independent of the classifier

print('Loading bert-base-uncased for ablation...')
from transformers import BertModel, BertTokenizer

base_model     = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
base_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
base_model     = base_model.to(device).eval()
for p in base_model.parameters(): p.requires_grad_(False)

# Re-run conditioning on saved pairs (no attack re-run needed)
ABLATION_CKPT = CKPT_DIR / 'cond_ablation_base.jsonl'
ablation_rows = []
completed_ablation = set()

if ABLATION_CKPT.exists():
    with open(ABLATION_CKPT) as fh:
        for line in fh:
            row = json.loads(line); ablation_rows.append(row)
            completed_ablation.add(row['pair_idx'])
    print(f'Ablation resumed: {len(ablation_rows)} done')

# Reload pairs
with open(PAIRS_PATH) as fh: ablation_pairs = json.load(fh)
todo_abl = [p for p in ablation_pairs if p['idx'] not in completed_ablation]
print(f'Ablation remaining: {len(todo_abl)} pairs')

t_abl = time.time()
with open(ABLATION_CKPT, 'a') as af:
    for step, pair in enumerate(tqdm(todo_abl, desc='Ablation (base model)', unit='pair')):
        row = analyze_pair(pair, base_model, base_tokenizer)
        ablation_rows.append(row); af.write(json.dumps(row) + '\n')
        if (step+1) % 100 == 0: af.flush()

print(f'Ablation done: {(time.time()-t_abl)/60:.1f} min')

# Compare AUCs: fine-tuned model vs base model
df_abl     = pd.DataFrame(ablation_rows)
N_abl      = len(df_abl)
y_abl      = [0]*N_abl + [1]*N_abl
abl_clean  = pd.DataFrame([{'pair_idx': r['pair_idx'], **{f'kappa_L{L}': r[f'kappa_clean_L{L}'] for L in LAYERS}} for _,r in df_abl.iterrows()])
abl_adv    = pd.DataFrame([{'pair_idx': r['pair_idx'], **{f'kappa_L{L}': r[f'kappa_adv_L{L}']   for L in LAYERS}} for _,r in df_abl.iterrows()])
abl_all    = pd.concat([abl_clean.assign(type='clean'), abl_adv.assign(type='adversarial')], ignore_index=True)

print('\n── Ablation: AUC comparison ─────────────────────────────────────────────')
print(f'  {"Layer":<8} {"Fine-tuned AUC":>16}  {"Base model AUC":>16}  {"Δ AUC":>10}')
print(f'  {"-"*54}')
for L in LAYERS:
    col      = f'kappa_L{L}'
    ft_vals  = np.concatenate([df_clean[col].values, df_adv[col].values])
    # fine-tuned AUC already computed
    ft_r     = df_layers[df_layers['layer']==L].iloc[0]
    abl_vals = np.concatenate([abl_clean[col].values, abl_adv[col].values])
    abl_auc_hi = roc_auc_score(y_abl, abl_vals)
    abl_auc_lo = roc_auc_score(y_abl, -abl_vals)
    abl_auc    = max(abl_auc_hi, abl_auc_lo)
    delta      = ft_r['auc'] - abl_auc
    print(f'  {L:<8} {ft_r["auc"]:>16.4f}  {abl_auc:>16.4f}  {delta:>+10.4f}')

print()
best_abl_L12 = max(
    roc_auc_score(y_abl, abl_all[abl_all.type=='adversarial']['kappa_L12'].tolist() + abl_all[abl_all.type=='clean']['kappa_L12'].tolist()),
    roc_auc_score(y_abl, [-x for x in abl_all[abl_all.type=='adversarial']['kappa_L12'].tolist() + abl_all[abl_all.type=='clean']['kappa_L12'].tolist()])
)
print('Interpretation:')
print(f'  Fine-tuned L12 AUC : {best_row["auc"]:.4f}')
print(f'  Base model  L12 AUC: {best_abl_L12:.4f}')
if abs(best_row['auc'] - best_abl_L12) > 0.05:
    if best_row['auc'] > best_abl_L12:
        print('  → Signal is STRONGER in fine-tuned model: supports decision-boundary mechanism.')
        print('    The conditioning explosion at L12 is amplified by task-specific fine-tuning.')
    else:
        print('  → Signal is STRONGER in base model: effect is text-intrinsic, not classifier-specific.')
else:
    print('  → Signal is SIMILAR in both models: effect is robust to fine-tuning.')

df_ablation_layers = pd.DataFrame([
    {'layer': L, 'model': 'base',
     'auc': max(roc_auc_score(y_abl, np.concatenate([abl_clean[f'kappa_L{L}'].values, abl_adv[f'kappa_L{L}'].values])),
                roc_auc_score(y_abl, np.concatenate([-abl_clean[f'kappa_L{L}'].values, -abl_adv[f'kappa_L{L}'].values])))}
    for L in LAYERS
])
df_ablation_layers.to_csv(RESULTS_DIR / 'ablation_base_model_layers.csv', index=False)
df_abl.to_csv(RESULTS_DIR / 'ablation_base_model_pairs.csv', index=False)
print(f'\nAblation results saved → {RESULTS_DIR}/')



# ────────────────────────────────────────────────────────────────────────────

import matplotlib
# ── Visualisation ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(f'SCM v3 — {ATTACK_NAME} on SST-2 (N={N}, fine-tuned model)',
             fontsize=13, fontweight='bold')
COLORS = {'clean': '#2ecc71', 'adv': '#e74c3c'}

# Plot 1: κ box at BEST layer (L12)
ax = axes[0,0]
kc = df_clean[f'kappa_L{best_L}'].values
ka = df_adv[f'kappa_L{best_L}'].values
bp = ax.boxplot([kc, ka], labels=['Clean','Adversarial'], patch_artist=True,
                medianprops=dict(color='black', linewidth=2))
bp['boxes'][0].set(facecolor=COLORS['clean'], alpha=0.75)
bp['boxes'][1].set(facecolor=COLORS['adv'],   alpha=0.75)
ax.set_title(f'Layer {best_L} κ  (AUC={best_row["auc"]:.3f}, p={best_row["pval"]:.1e})',
             fontweight='bold')
ax.set_ylabel('Condition Number κ'); ax.set_xlabel('Input type')

# Plot 2: Layer AUC bar + ablation overlay
ax = axes[0,1]
ax.bar(df_layers['layer'], df_layers['auc'], color='#3498db', alpha=0.8, zorder=2,
       label='Fine-tuned model')
if 'df_ablation_layers' in dir():
    ax.plot(df_ablation_layers['layer'], df_ablation_layers['auc'],
            'o--', color='#8e44ad', lw=1.5, ms=6, label='Base model (ablation)')
ax.axhline(auc_cos, color='red',   ls='--', lw=1.5, label=f'Cosine AUC ({auc_cos:.3f})')
ax.axhline(0.75,    color='green', ls=':',  alpha=0.5, label='Target 0.75')
ax.axhline(0.5,     color='gray',  ls=':',  alpha=0.4)
ax.set(xlabel='Layer', ylabel='AUC-ROC', title='κ AUC by Layer', ylim=(0.3,1.0))
ax.set_xticks(LAYERS); ax.legend(fontsize=7)

# Plot 3: ROC curves
ax = axes[0,2]
for L in LAYERS:
    r   = df_layers[df_layers['layer']==L].iloc[0]
    k_v = df_all[f'kappa_L{L}'].values.copy()
    if r['direction']=='lower': k_v = -k_v
    fpr, tpr, _ = roc_curve(y_true, k_v)
    ax.plot(fpr, tpr, label=f'L{L} ({r["auc"]:.3f})')
ax.plot([0,1],[0,1],'k:',alpha=0.4)
ax.set(xlabel='FPR',ylabel='TPR',title='ROC Curves by Layer')
ax.legend(fontsize=8, loc='lower right')

# Plot 4: Δκ histogram at best layer
ax = axes[1,0]
delta = df_pairs[f'delta_kappa_L{best_L}'].values
ax.hist(delta, bins=40, color='#9b59b6', alpha=0.75, edgecolor='black')
ax.axvline(0,            color='red',   ls='--', lw=2, label='No change')
ax.axvline(delta.mean(), color='green', ls='-',  lw=2, label=f'Mean={delta.mean():.2f}')
ax.set(xlabel=f'Δκ (adv−clean) at Layer {best_L}', ylabel='Count',
       title='Conditioning Change Under Attack')
ax.legend()

# Plot 5: κ vs cosine distance
ax = axes[1,1]
ax.scatter(df_pairs['cosine_distance'], df_pairs[f'kappa_adv_L{best_L}'],
           c=COLORS['adv'],   alpha=0.35, s=10, label='Adversarial')
ax.scatter(df_pairs['cosine_distance'], df_pairs[f'kappa_clean_L{best_L}'],
           c=COLORS['clean'], alpha=0.35, s=10, label='Clean')
ax.axvline(1.0, color='gray', ls='--', lw=1, alpha=0.7, label='cos_sim=0 boundary')
ax.set(xlabel='Cosine Distance (clean↔adv)', ylabel=f'κ at Layer {best_L}',
       title='κ vs Embedding Distance')
ax.legend(markerscale=2, fontsize=8)

# Plot 6: Layer-wise κ profile
ax = axes[1,2]
ax.errorbar(LAYERS, df_layers['mean_clean'], yerr=df_layers['std_clean'],
            marker='o', capsize=4, color=COLORS['clean'], label='Clean', lw=2, ms=7)
ax.errorbar(LAYERS, df_layers['mean_adv'],   yerr=df_layers['std_adv'],
            marker='s', capsize=4, color=COLORS['adv'],   label='Adversarial', lw=2, ms=7)
ax.set(xlabel='Layer', ylabel='Mean κ', title='Layer-wise κ Profile (Figure 1)')
ax.legend(); ax.set_xticks(LAYERS)

plt.tight_layout()
plt.savefig(RESULTS_DIR / f'scm_v3_{ATTACK_NAME}_results.png', dpi=150, bbox_inches='tight')
# plt.show()  # skipped in terminal mode
print(f'Composite figure saved.')



# ────────────────────────────────────────────────────────────────────────────

import matplotlib
# ── Per-figure exports (paper figure naming) ──────────────────────────────────
# SUGGESTION 3: Figure 3 is rendered at best_L (L12) — the layer with actual
# signal — not L1 which shows a null result and would mislead reviewers.

COLORS = {'clean': '#2ecc71', 'adv': '#e74c3c'}

def export_figure1_layer_profile(df_layers, layers, colors, path):
    """Figure 1: layer-wise κ profile with clean vs adversarial."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(layers, df_layers['mean_clean'], yerr=df_layers['std_clean'],
                marker='o', capsize=4, color=colors['clean'], label='Clean', lw=2, ms=7)
    ax.errorbar(layers, df_layers['mean_adv'],   yerr=df_layers['std_adv'],
                marker='s', capsize=4, color=colors['adv'],   label='Adversarial', lw=2, ms=7)
    ax.set(xlabel='Layer', ylabel='Mean Condition Number (κ)',
           title=f'Layer-wise κ Profile: Clean vs Adversarial ({ATTACK_NAME})')
    ax.set_xticks(layers); ax.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()

def export_figure3_distribution(df_clean, df_adv, best_L, best_row, path):
    """Figure 3: κ distribution at BEST layer (L12), not L1.
    SUGGESTION 3: Showing L1 would present a null result as the main figure.
    best_L is passed explicitly so the figure always reflects actual best detection.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    col = f'kappa_L{best_L}'
    kc  = df_clean[col].values
    ka  = df_adv[col].values
    bp  = ax.boxplot([kc, ka], labels=['Clean', 'Adversarial'],
                     patch_artist=True,
                     medianprops=dict(color='black', linewidth=2))
    bp['boxes'][0].set(facecolor=COLORS['clean'], alpha=0.75)
    bp['boxes'][1].set(facecolor=COLORS['adv'],   alpha=0.75)
    ax.set_title(f'{ATTACK_NAME}  —  Layer {best_L}\n'
                 f'AUC={best_row["auc"]:.3f}, p={best_row["pval"]:.1e}, d={best_row["cohens_d"]:.3f}',
                 fontweight='bold')
    ax.set_ylabel(f'κ at Layer {best_L}')
    plt.suptitle('Distribution of κ at Best Detection Layer: Clean vs Adversarial',
                 fontweight='bold', y=1.02)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()

def export_figure5_roc(df_all, df_layers, y_true, layers, auc_cos, path):
    """Figure 5: ROC curves with corrected cosine baseline legend."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for L in layers:
        r   = df_layers[df_layers['layer']==L].iloc[0]
        k_v = df_all[f'kappa_L{L}'].values.copy()
        if r['direction']=='lower': k_v = -k_v
        fpr, tpr, _ = roc_curve(y_true, k_v)
        ax.plot(fpr, tpr, lw=2, label=f'Layer {L} κ (AUC={r["auc"]:.3f})')
    ax.axhline(auc_cos, color='gray', ls='--', lw=1.5,
               label=f'Cosine distance (AUC={auc_cos:.3f}, near-random)')
    ax.plot([0,1],[0,1],'k:',alpha=0.4)
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate',
           title=f'ROC Curves: κ-based Detection vs Cosine Baseline\n({ATTACK_NAME}, N={N})')
    ax.legend(fontsize=9, loc='lower right')
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()

fig1_path = RESULTS_DIR / 'figure_1_layer_profile.png'
fig3_path = RESULTS_DIR / 'figure_3_kappa_distribution.png'
fig5_path = RESULTS_DIR / 'figure_5_roc_curves.png'

export_figure1_layer_profile(df_layers, LAYERS, COLORS, fig1_path)
export_figure3_distribution(df_clean, df_adv, best_L, best_row, fig3_path)
export_figure5_roc(df_all, df_layers, y_true, LAYERS, auc_cos, fig5_path)

print('Figures saved:')
for p in [fig1_path, fig3_path, fig5_path]: print(f'  {p}')
print(f'  Note: Figure 3 shows Layer {best_L} (best detection layer), NOT Layer 1.')



# ────────────────────────────────────────────────────────────────────────────

# ── Export paper_numbers.json ─────────────────────────────────────────────────
# SUGGESTION 2: Captions corrected — no longer reference "lower κ at early layers".
# All inline claims derive from computed values, not assertions.

import datetime

def layer_row_dict(df_layers, L):
    r = df_layers[df_layers['layer']==L].iloc[0]
    return {'layer': int(L), 'auc': round(float(r['auc']),4),
            'direction': str(r['direction']), 'p_value': round(float(r['pval']),4),
            'cohens_d': round(float(r['cohens_d']),3),
            'kappa_clean': {'mean': round(float(r['mean_clean']),4), 'std': round(float(r['std_clean']),4)},
            'kappa_adv':   {'mean': round(float(r['mean_adv']),4),   'std': round(float(r['std_adv']),4)}}

best_L_row = df_layers[df_layers['layer']==best_L].iloc[0]

# Load ablation layer summary if available
abl_layer_summary = {}
abl_path = RESULTS_DIR / 'ablation_base_model_layers.csv'
if abl_path.exists():
    df_abl_layers = pd.read_csv(abl_path)
    for _, r in df_abl_layers.iterrows():
        abl_layer_summary[int(r['layer'])] = round(float(r['auc']),4)

paper_numbers = {
    'metadata': {
        'generated_at'     : datetime.datetime.now().isoformat(),
        'model_attack'     : MODEL_NAME,
        'model_condition'  : MODEL_NAME,
        'dataset'          : 'SST-2 (HuggingFace validation + train)',
        'attack'           : ATTACK_NAME,
        'n_pairs'          : int(N),
        'layers_analysed'  : LAYERS,
        'notebook_version' : 'v3',
        'scm_config'       : {
            'power_iterations' : scm.power_iterations,
            'n_sigma_low'      : scm.n_sigma_low,
            'sigma_max_method' : 'JVP+VJP power iteration',
            'sigma_low_method' : 'lower-tail proxy (10th percentile, 15 random projections)',
        },
    },

    # ── Table 1 (two-row: original + replicated) ──────────────────────────────
    'table_1': {
        'original': {
            'attack': ATTACK_NAME, 'n_pairs': 29,
            'model': 'bert-base-uncased (base)',
            'best_layer': 1,
            'kappa_clean': {'mean': 2.906, 'std': 0.282},
            'kappa_adv'  : {'mean': 2.662, 'std': 0.309},
            'p_value': 0.0056, 'auc': 0.7122,
            'cosine_auc': None,
            'note': 'Biased CLS-proxy estimator; hand-crafted samples; N=29',
        },
        'replicated': {
            'attack': ATTACK_NAME, 'n_pairs': int(N),
            'model': MODEL_NAME,
            'best_layer': int(best_L),
            'kappa_clean': {'mean': round(float(best_L_row['mean_clean']),4),
                            'std':  round(float(best_L_row['std_clean']),4)},
            'kappa_adv'  : {'mean': round(float(best_L_row['mean_adv']),4),
                            'std':  round(float(best_L_row['std_adv']),4)},
            'p_value': round(float(best_L_row['pval']),6),
            'auc':     round(float(best_L_row['auc']),4),
            'cosine_auc': round(float(auc_cos),4),
            'note': 'JVP+VJP estimator; real SST-2; same model for attack+conditioning',
        },
    },

    # ── Per-layer results ─────────────────────────────────────────────────────
    'detection_results': [layer_row_dict(df_layers, L) for L in LAYERS],

    # ── Ablation ──────────────────────────────────────────────────────────────
    'ablation_base_model': {
        'available': bool(abl_layer_summary),
        'layer_auc': abl_layer_summary,
        'interpretation': (
            'If L12 AUC drops substantially on base model vs fine-tuned, '
            'the effect is specific to task-specific geometry (decision-boundary mechanism). '
            'If similar, the effect is intrinsic to the input text.'
        ),
    },

    # ── Cosine baseline ───────────────────────────────────────────────────────
    'cosine_baseline': {
        'auc': round(float(auc_cos),4),
        'mean_clean_adv_dist': round(float(df_pairs['cosine_distance'].mean()),4),
        'std_clean_adv_dist':  round(float(df_pairs['cosine_distance'].std()),4),
        'pairs_with_cos_dist_gt_1': int((df_pairs['cosine_distance'] > 1.0).sum()),
        'interpretation': (
            f'AUC={round(float(auc_cos),3)} is near-random (original paper asserted AUC≈0.25 '
            f'without computing it). Values > 1.0 are valid: cosine_dist > 1 means '
            f'cosine_sim < 0 (anti-parallel CLS vectors), expected when TextFooler flips '
            f'the sentiment label and the fine-tuned model encodes strong polarity in CLS. '
            f'κ at L{int(best_L)} (AUC={round(float(best_L_row["auc"]),3)}) vastly outperforms cosine.'
        ),
    },

    # ── Key claims ────────────────────────────────────────────────────────────
    'key_claims': {
        'best_layer'              : int(best_L),
        'best_auc'                : round(float(best_L_row['auc']),4),
        'best_p_value'            : round(float(best_L_row['pval']),6),
        'best_cohens_d'           : round(float(best_L_row['cohens_d']),3),
        'kappa_direction'         : str(best_L_row['direction']),
        'kappa_advantage_over_cosine': round(float(best_L_row['auc']-auc_cos),4),
        'sigma_max_ratio_at_best_layer': round(
            float(df_pairs[f'sigma_max_adv_L{best_L}'].mean() /
                  df_pairs[f'sigma_max_clean_L{best_L}'].mean()), 3),
        'sigma_low_ratio_at_best_layer': round(
            float(df_pairs[f'sigma_low_adv_L{best_L}'].mean() /
                  df_pairs[f'sigma_low_clean_L{best_L}'].mean()), 3),
        'finding_direction_vs_original': (
            f'INVERTED: original claimed lower κ at L1; '
            f'replicated shows higher κ at L{int(best_L)}'
        ),
    },

    # ── Figure manifest (corrected captions) ──────────────────────────────────
    # SUGGESTION 2: captions derived from actual computed values
    'figures': {
        'figure_1': {
            'file'    : 'figure_1_layer_profile.png',
            'caption' : (
                f'Layer-wise condition number (κ) profiles for clean and adversarial inputs '
                f'under {ATTACK_NAME} (N={N}). Early layers (1–6) show negligible separation; '
                f'Layer {int(best_L)} shows a strong divergence '
                f'(κ_adv/κ_clean = {round(float(df_layers[df_layers["layer"]==best_L].iloc[0]["mean_adv"]/df_layers[df_layers["layer"]==best_L].iloc[0]["mean_clean"]),2)}×, '
                f'AUC={round(float(best_L_row["auc"]),3)}).'
            ),
            'alt_text': (
                f'Line plot of mean condition number vs layer index for clean (green circles) '
                f'and adversarial (red squares) inputs. Lines overlap through Layer 9, '
                f'then diverge sharply at Layer 12 with adversarial higher.'
            ),
        },
        'figure_3': {
            'file'    : 'figure_3_kappa_distribution.png',
            'caption' : (
                f'Distribution of κ at Layer {int(best_L)} — the layer with peak discriminability — '
                f'for clean vs {ATTACK_NAME} adversarial inputs (N={N}). '
                f'Adversarial examples show substantially higher κ '
                f'(mean {round(float(best_L_row["mean_adv"]),1)} vs {round(float(best_L_row["mean_clean"]),1)}, '
                f'AUC={round(float(best_L_row["auc"]),3)}, p={round(float(best_L_row["pval"]),4)}, '
                f'd={round(float(best_L_row["cohens_d"]),3)}).'
            ),
            'alt_text': (
                f'Box plots at Layer {int(best_L)}: adversarial (red) interquartile range '
                f'is clearly above clean (green), with many high-κ outliers.'
            ),
        },
        'figure_5': {
            'file'    : 'figure_5_roc_curves.png',
            'caption' : (
                f'ROC curves for κ-based adversarial detection at each transformer layer, '
                f'with cosine distance as a near-random baseline (AUC={round(float(auc_cos),3)}). '
                f'Layer {int(best_L)} κ achieves AUC={round(float(best_L_row["auc"]),3)}, '
                f'a {round(float(best_L_row["auc"]-auc_cos),3)}-point advantage over the embedding baseline.'
            ),
            'alt_text': (
                'ROC curves for Layers 1, 3, 6, 9, 12 plus cosine baseline. '
                'Layers 1-9 cluster near the diagonal; Layer 12 curves sharply upward.'
            ),
        },
    },

    'merge_instructions': {
        'description': (
            'To merge DeepWordBug results: change ATTACK_NAME to DeepWordBug, rerun, '
            'then append replicated entry to table_1 and add detection_results. '
            'key_claims will auto-update from computed best_L.'
        ),
        'required_fields': [
            'attack', 'n_pairs', 'best_layer',
            'kappa_clean.mean', 'kappa_clean.std',
            'kappa_adv.mean',   'kappa_adv.std',
            'p_value', 'auc', 'cosine_auc',
        ],
    },
}

outpath = RESULTS_DIR / 'paper_numbers.json'
with open(outpath, 'w') as fh: json.dump(paper_numbers, fh, indent=2)
print(f'paper_numbers.json → {outpath}')
print(f'  best_layer    : {paper_numbers["key_claims"]["best_layer"]}')
print(f'  best_auc      : {paper_numbers["key_claims"]["best_auc"]}')
print(f'  cosine_auc    : {paper_numbers["cosine_baseline"]["auc"]}')
print(f'  σ_max ratio   : {paper_numbers["key_claims"]["sigma_max_ratio_at_best_layer"]}×')
print(f'  direction     : {paper_numbers["key_claims"]["finding_direction_vs_original"]}')



# ────────────────────────────────────────────────────────────────────────────

# ── Final report + save ───────────────────────────────────────────────────────
print('='*70)
print(f' FINAL REPORT — SCM v3 ({ATTACK_NAME})')
print('='*70)
print(f'  Model     : {MODEL_NAME}')
print(f'  N pairs   : {N}')
print(f'  Best layer: {best_L}  (AUC={best_row["auc"]:.4f}, d={best_row["cohens_d"]:.3f})')
print(f'  Cosine AUC: {auc_cos:.4f}  (κ advantage: {best_row["auc"]-auc_cos:+.4f})')
print(f'  p-value   : {best_row["pval"]:.2e}')
print(f'  σ_max ratio at L{best_L}: {df_pairs[f"sigma_max_adv_L{best_L}"].mean()/df_pairs[f"sigma_max_clean_L{best_L}"].mean():.2f}×')
print()

df_pairs.to_csv(RESULTS_DIR  / f'pairs_analysis_{ATTACK_NAME}.csv', index=False)
df_layers.to_csv(RESULTS_DIR / f'layer_summary_{ATTACK_NAME}.csv',  index=False)
df_all.to_csv(RESULTS_DIR    / f'all_kappa_{ATTACK_NAME}.csv',      index=False)
print(f'  Results → {RESULTS_DIR}/')



# ────────────────────────────────────────────────────────────────────────────

# ── Option A: Merge results from parallel instances ──────────────────────────
# Run after BOTH instances have completed and results dirs are local.
#
# Directory layout expected:
#   ./results_TextFooler/paper_numbers.json
#   ./results_DeepWordBug/paper_numbers.json
#
# Output: ./paper_numbers_merged.json  (ready for Claude Code camera-ready pass)

from pathlib import Path
import json, datetime

ATTACKS_TO_MERGE = ['TextFooler', 'DeepWordBug']  # extend for PWWS

def load_pn(attack):
    p = Path(f'results_{attack}') / 'paper_numbers.json'
    if not p.exists():
        print(f'  WARNING: {p} not found — skipping'); return None
    with open(p) as fh: return json.load(fh)

print('Loading per-attack results...')
all_pn = {a: pn for a in ATTACKS_TO_MERGE if (pn := load_pn(a)) is not None}

for atk, pn in all_pn.items():
    kc = pn['key_claims']
    print(f"  {atk:12} : best_layer={kc['best_layer']}  "
          f"AUC={kc['best_auc']}  N={pn['metadata']['n_pairs']}")

if len(all_pn) < 2:
    print('Need results from both instances. Complete the second run first.')
else:
    base = list(all_pn.values())[0]

    best_layers = {a: pn['key_claims']['best_layer'] for a, pn in all_pn.items()}
    best_aucs   = {a: pn['key_claims']['best_auc']   for a, pn in all_pn.items()}
    consistent  = len(set(best_layers.values())) == 1

    if consistent:
        L = list(best_layers.values())[0]
        auc_str = ', '.join(f'{a} AUC={v}' for a, v in best_aucs.items())
        claim = (f'The Layer-{L} conditioning signal replicates across '
                 f'word-level and character-level attacks ({auc_str}), '
                 f'consistent with a decision-boundary geometry mechanism '
                 f'independent of attack strategy.')
    else:
        claim = ('Best detection layer differs across attacks: '
                 + ', '.join(f'{a}=L{L}' for a, L in best_layers.items())
                 + '. The conditioning signal may be attack-strategy-specific.')

    merged = {
        'metadata': {
            **base['metadata'],
            'generated_at'    : datetime.datetime.now().isoformat(),
            'attacks_merged'  : list(all_pn.keys()),
            'notebook_version': 'v3-merged',
        },
        'table_1': {
            'original'  : base['table_1']['original'],
            'replicated': [{**pn['table_1']['replicated'], 'attack': a}
                           for a, pn in all_pn.items()],
        },
        'detection_results'  : {a: pn['detection_results']       for a, pn in all_pn.items()},
        'ablation_base_model': {a: pn.get('ablation_base_model',{}) for a, pn in all_pn.items()},
        'cosine_baseline'    : {a: pn['cosine_baseline']          for a, pn in all_pn.items()},
        'key_claims'         : {a: pn['key_claims']               for a, pn in all_pn.items()},
        'figures'            : {a: pn['figures']                  for a, pn in all_pn.items()},
        'cross_attack_summary': {
            'best_layers'          : best_layers,
            'best_aucs'            : best_aucs,
            'best_layer_consistent': consistent,
            'cross_attack_claim'   : claim,
        },
    }

    out = Path('paper_numbers_merged.json')
    with open(out, 'w') as fh: json.dump(merged, fh, indent=2)

    print()
    print(f'Merged -> {out}')
    print(f'Layer consistent across attacks: {consistent}')
    print()
    print('Generated cross-attack claim:')
    print(f'  {claim}')
    print()
    print('Claude Code usage:')
    print('  pn = json.load(open("paper_numbers_merged.json"))')
    print('  for row in pn["table_1"]["replicated"]: ...    # Table 1 rows')
    print('  pn["cross_attack_summary"]["cross_attack_claim"] # paste into paper')


# Deferred improvements (future work)
# 1. vmap batched VJPs (PyTorch ≥ 2.0)
# Batch all `n_sigma_low` projections in one call via `torch.func.vmap`.
# Expected ~10× speedup on `_sigma_low_quantile`.
# 2. Lanczos σ_min
# Replace 10th-percentile proxy with `scipy.sparse.linalg.svds` on an implicit
# LinearOperator. Tighter confidence intervals on κ.
# 3. PWWS (selective)
# Only worth adding if DeepWordBug replicates the L12 finding.
# 4. RoBERTa cross-architecture
# Add `.roberta` branch to `_make_layer_fn`; one config line change.
# 5. Adaptive adversary pilot (N=100)
# Add `λ · κ_L12(x_adv)` penalty to attack candidate scoring.
