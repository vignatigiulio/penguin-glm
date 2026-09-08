#!/usr/bin/env python3
"""
inspect_embeddings.py -- PUNTO 2: la 519M e' "stabile" (tesi) o ha "instabilita'
residua" (README)?

Le due affermazioni misurano cose diverse: un valore puo' essere FINITO ma enorme,
e finire clampato senza essere mai stato NaN/Inf. Questo script separa le due cose:

    non-finiti  -> NaN o Inf veri e propri        (la tesi dice: zero)
    clamp       -> valori azzerati/saturati       (il README dice: 2-3%)

USO
    # 1. ispezione di uno o piu' archivi
    python inspect_embeddings.py $PENGUIN_ROOT/out/embeddings/*519*.npz

    # 2. confronto clamp vs no-clamp SULLE METRICHE (non sui vettori)
    python inspect_embeddings.py emb_clamp.npz emb_noclamp.npz --compare

    # 3. come sopra, aggiungendo la sonda lineare (piu' lento, ~1-3 min)
    python inspect_embeddings.py emb_clamp.npz emb_noclamp.npz --compare --probe

NOTA sul confronto: riproduce il protocollo della tesi solo approssimativamente.
Cio' che conta e' il DELTA fra i due archivi, non il valore assoluto.
"""

import argparse
import glob
import os
import sys

import numpy as np

FLOAT_KEYS = ("embedding", "emb", "x", "features", "hidden", "repr")
LABEL_KEYS = ("label", "labels", "y", "class", "classe")
GC_KEYS = ("gc", "gc_content", "gc_frac")
SPLIT_KEYS = ("split", "partition", "set")


def pick(z, candidates):
    """Trova la prima chiave che assomiglia a una di quelle candidate."""
    keys = list(z.keys())
    for c in candidates:
        for k in keys:
            if k.lower() == c:
                return k
    for c in candidates:
        for k in keys:
            if c in k.lower():
                return k
    return None


def biggest_float(z):
    """L'array float 2-D piu' grande: quasi sempre e' la matrice di embedding."""
    best, best_size = None, -1
    for k in z.keys():
        try:
            a = z[k]
        except Exception:
            continue
        if getattr(a, "ndim", 0) == 2 and np.issubdtype(a.dtype, np.floating):
            if a.size > best_size:
                best, best_size = k, a.size
    return best


def describe(path, deep=True):
    print(f"\n{'='*92}\n{path}\n{'='*92}")
    if not os.path.isfile(path):
        print("  FILE ASSENTE")
        return None
    try:
        z = np.load(path, allow_pickle=True)
    except Exception as e:
        print(f"  ERRORE nel caricamento: {e}")
        return None

    print(f"  chiavi: {list(z.keys())}\n")

    # -- metadati: spesso salvati come dict in un array 0-d ------------------
    meta_found = False
    for k in z.keys():
        kl = k.lower()
        if any(t in kl for t in ("clamp", "meta", "config", "cfg", "info", "sdp", "amp")):
            try:
                v = z[k]
                v = v.item() if getattr(v, "ndim", 1) == 0 else v
            except Exception:
                v = "<illeggibile>"
            print(f"  [META] {k} = {v}")
            meta_found = True
    if not meta_found:
        print("  [META] nessuna chiave di metadati riconosciuta "
              "(cerca 'clamp' nel sorgente di estrazione)")

    ek = pick(z, FLOAT_KEYS) or biggest_float(z)
    if ek is None:
        print("\n  nessuna matrice float trovata")
        return None

    X = z[ek]
    print(f"\n  matrice: '{ek}'  shape={X.shape}  dtype={X.dtype}")

    if not np.issubdtype(X.dtype, np.floating):
        print("  (non float: salto le statistiche numeriche)")
        return dict(path=path, key=ek, X=None, z=z)

    Xf = X.astype(np.float64, copy=False)
    finite = np.isfinite(Xf)
    n_nonfinite = int((~finite).sum())
    n_nan = int(np.isnan(Xf).sum())
    n_inf = int(np.isinf(Xf).sum())
    tot = Xf.size

    print(f"\n  --- NON-FINITI (l'affermazione della tesi) ---")
    print(f"  NaN            : {n_nan}")
    print(f"  Inf            : {n_inf}")
    print(f"  totale         : {n_nonfinite}  su {tot}  ({100*n_nonfinite/tot:.4f}%)")
    print(f"  righe colpite  : {int((~finite).any(axis=1).sum())} su {X.shape[0]}")

    if n_nonfinite == 0:
        print("  -> coerente con la tesi: nessun valore non finito.")
    else:
        print("  -> IN CONTRASTO con la tesi: ci sono valori non finiti.")

    if deep and finite.any():
        vals = Xf[finite]
        n_zero = int((Xf == 0).sum())
        rows_with_zero = int((Xf == 0).any(axis=1).sum())
        print(f"\n  --- IMPRONTA DEL CLAMP (l'affermazione del README) ---")
        print(f"  zeri esatti    : {n_zero}  ({100*n_zero/tot:.3f}% dei valori)")
        print(f"  righe con zeri : {rows_with_zero}  ({100*rows_with_zero/X.shape[0]:.2f}%)")
        print(f"  |x| max        : {np.abs(vals).max():.4e}")
        print(f"  |x| p99.9      : {np.percentile(np.abs(vals), 99.9):.4e}")
        print(f"  media / dev.st : {vals.mean():.4e} / {vals.std():.4e}")
        if 0.005 <= n_zero / tot <= 0.05:
            print("  -> la frazione di zeri e' nell'ordine del 2-3% citato dal README:")
            print("     compatibile con un clamp che azzera, NON con valori non finiti.")
        elif n_zero == 0:
            print("  -> nessuno zero esatto: nessuna traccia di clamp in questo archivio.")

    return dict(path=path, key=ek, X=X, z=z)


# --------------------------------------------------- confronto sulle metriche ---

def metrics(X, z, probe=False, seed=0):
    """NMI (k-means, k=4) rispetto a classi e GC; opzionale MCC da sonda lineare."""
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import normalized_mutual_info_score
    except ImportError:
        print("  sklearn non disponibile: salto le metriche")
        return {}

    out = {}
    Xc = np.nan_to_num(X.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    Xs = StandardScaler().fit_transform(Xc)

    km = KMeans(n_clusters=4, n_init=10, random_state=seed).fit(Xs)
    lab_k = pick(z, LABEL_KEYS)
    gc_k = pick(z, GC_KEYS)

    if lab_k is not None:
        y = np.asarray(z[lab_k]).ravel()
        out["NMI_funzione"] = normalized_mutual_info_score(y, km.labels_)
    if gc_k is not None:
        gc = np.asarray(z[gc_k]).ravel().astype(float)
        gcb = np.digitize(gc, np.quantile(gc, [0.25, 0.5, 0.75]))
        out["NMI_GC"] = normalized_mutual_info_score(gcb, km.labels_)
    if "NMI_funzione" in out and out.get("NMI_GC"):
        out["rapporto_f_GC"] = out["NMI_funzione"] / out["NMI_GC"]

    if probe and lab_k is not None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import matthews_corrcoef
        y = np.asarray(z[lab_k]).ravel()
        sp_k = pick(z, SPLIT_KEYS)
        if sp_k is not None:
            sp = np.asarray(z[sp_k]).ravel().astype(str)
            tr = np.char.startswith(np.char.lower(sp), "tr")
            te = np.char.startswith(np.char.lower(sp), "te")
            if te.sum() == 0:
                te = ~tr
        else:
            rng = np.random.default_rng(seed)
            idx = rng.permutation(len(y))
            cut = int(0.8 * len(y))
            tr = np.zeros(len(y), bool); tr[idx[:cut]] = True
            te = ~tr
        clf = LogisticRegression(max_iter=2000).fit(Xs[tr], y[tr])
        out["MCC_sonda_lineare"] = matthews_corrcoef(y[te], clf.predict(Xs[te]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="file .npz (accetta glob della shell)")
    ap.add_argument("--compare", action="store_true",
                    help="confronta i primi due archivi sulle METRICHE")
    ap.add_argument("--probe", action="store_true",
                    help="aggiunge la sonda lineare al confronto (piu' lento)")
    ap.add_argument("--quick", action="store_true", help="salta le statistiche pesanti")
    args = ap.parse_args()

    paths = []
    for p in args.paths:
        paths.extend(sorted(glob.glob(p)) or [p])

    loaded = [describe(p, deep=not args.quick) for p in paths]
    loaded = [d for d in loaded if d and d.get("X") is not None]

    if args.compare:
        if len(loaded) < 2:
            sys.exit("\nservono due archivi leggibili per --compare")
        a, b = loaded[0], loaded[1]
        print(f"\n{'='*92}\nCONFRONTO SULLE METRICHE (cio' che regge l'argomento)\n{'='*92}")
        print(f"  A = {a['path']}\n  B = {b['path']}\n")

        if a["X"].shape == b["X"].shape:
            A = np.nan_to_num(a["X"].astype(np.float64))
            B = np.nan_to_num(b["X"].astype(np.float64))
            num = (A * B).sum(1)
            den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12
            cos = num / den
            print(f"  coseno per riga: min={cos.min():.6f}  medio={cos.mean():.6f}")
            rel = np.abs(A - B).sum() / (np.abs(A).sum() + 1e-12)
            print(f"  differenza relativa: {100*rel:.2f}%\n")
        else:
            print("  shape diverse: salto il confronto vettoriale\n")

        ma = metrics(a["X"], a["z"], probe=args.probe)
        mb = metrics(b["X"], b["z"], probe=args.probe)
        keys = sorted(set(ma) | set(mb))
        if keys:
            print(f"  {'metrica':<20} {'A':>10} {'B':>10} {'delta':>10}")
            for k in keys:
                va, vb = ma.get(k), mb.get(k)
                d = f"{vb-va:+.4f}" if (va is not None and vb is not None) else "-"
                print(f"  {k:<20} {va if va is None else f'{va:.4f}':>10} "
                      f"{vb if vb is None else f'{vb:.4f}':>10} {d:>10}")
            print("\n  Se i delta sono trascurabili, il clamp e' cosmetico e va")
            print("  dichiarato come diagnostica. Se si muovono, ogni risultato della")
            print("  519M deve riportare la clamp_fraction, come chiede il README.")


if __name__ == "__main__":
    main()
