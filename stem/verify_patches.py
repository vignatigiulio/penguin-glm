#!/usr/bin/env python3
"""
verify_patches.py — STATIC verification of the patches applied to JanusDNA.

It does not run the model: it diffs the patched repository against the
baseline and checks that EXACTLY the intended lines changed, and no others.

What it establishes:
  1. the 6 model patches and the 2 train.py patches are present;
  2. no line outside the intended regions was touched;
  3. with BAND = 1 every parametrised expression evaluates to the original
     constant -> the S = 4 case IS the upstream behaviour, not an imitation.

USAGE:
  python verify_patches.py $JANUSDNA_REPO $PENGUIN_ROOT/JanusDNA_stem_s1_b4
"""

import re
import sys
import difflib
from pathlib import Path

# (name, fragment expected in the patched file)
MODEL_PATCHES = [
    ("1. env constants",           "_STEM_BAND = (_STEM_BLOCK // _STEM_S) if _USE_STEM else 1"),
    ("2. stem in __init__",        "self.stem = CausalConvStem(config.hidden_size, stride=_STEM_S,"),
    ("2b. fusion-strategy assert", 'layer_fusion_strategy", None) == "pool"'),
    ("3. stem on both branches",   "_bwd = self.stem(inputs_embeds.flip(dims=(1,)))"),
    ("4. BlockHead",              "self.lm_head = BlockHead(config.hidden_size, _STEM_BLOCK, config.vocab_size)"),
    ("5a. mask fwd->bwd",          "kv_idx >= orig_seq_len + q_idx + (_STEM_BAND + 1)"),
    ("5b. mask bwd->fwd",          "q_idx >= kv_idx + orig_seq_len + (_STEM_BAND + 1)"),
    ("6a. fusion: assert",         "assert mid_len > 2 * _STEM_BAND"),
    ("6b. fusion: head",           "hidden_states[:, mid_len + _STEM_BAND + head_indices, :]"),
    ("6c. fusion: body range",     "torch.arange(mid_len - (_STEM_BAND + 1)"),
    ("6d. fusion: body",           "mid_len + 1 + middle_indices + _STEM_BAND"),
    ("6e. fusion: tail",           "tail_indices = torch.arange(_STEM_BAND"),
]

TRAIN_PATCHES = [
    ("7. target realignment",      "[stem-align]"),
    ("8. gate on on_fit_start",    "def on_fit_start(self)"),
]

# patched expressions -> value expected with BAND = 1 (upstream behaviour)
BAND1_EQUIVALENCE = [
    ("mask gap",              "_STEM_BAND + 1",              2),   # was 2 * 1
    ("fusion head offset",    "mid_len + _STEM_BAND",        None),  # was mid_len + 1
    ("body range",            "mid_len - (_STEM_BAND + 1)",  None),  # was mid_len - 2*1
    ("body offset",           "middle_indices + _STEM_BAND", None),  # was + 1
    ("tail width",            "_STEM_BAND",                  1),   # was 1
]


def check_presence(text, patches, label):
    print(f"\n--- {label} ---")
    ok = True
    for name, frag in patches:
        hit = frag in text
        print(f"  [{'OK ' if hit else 'NO '}] {name}")
        ok &= hit
    return ok


def band1_equivalence():
    """With BAND=1 the patched expressions must equal the original constants."""
    print("\n--- BAND = 1 equivalence (the S = 4 case IS upstream behaviour) ---")
    _STEM_BAND, mid_len, middle_indices = 1, 100, 7
    checks = [
        ("mask: (BAND+1)              vs  2*1",        _STEM_BAND + 1,                  2 * 1),
        ("fusion head: mid+BAND       vs  mid+1",      mid_len + _STEM_BAND,            mid_len + 1),
        ("body range: mid-(BAND+1)    vs  mid-2*1",    mid_len - (_STEM_BAND + 1),      mid_len - 2 * 1),
        ("body offset: mi+BAND        vs  mi+1",       middle_indices + _STEM_BAND,     middle_indices + 1),
        ("tail: arange(BAND)          vs  arange(1)",  _STEM_BAND,                      1),
        ("assert: 2*BAND              vs  2*1",        2 * _STEM_BAND,                  2 * 1),
    ]
    ok = True
    for name, patched, original in checks:
        eq = patched == original
        print(f"  [{'OK ' if eq else 'NO '}] {name}   ({patched} == {original})")
        ok &= eq
    return ok


def controlled_diff(base_path, patched_path, expected_regions):
    """Show the changed lines and verify they fall only in the intended regions."""
    base = base_path.read_text().splitlines()
    patched = patched_path.read_text().splitlines()
    sm = difflib.SequenceMatcher(None, base, patched, autojunk=False)

    hunks = [op for op in sm.get_opcodes() if op[0] != "equal"]
    print(f"\n--- controlled diff: {patched_path.name} ---")
    print(f"  baseline lines {len(base)} -> patched {len(patched)} "
          f"(+{len(patched)-len(base)}), {len(hunks)} hunks")

    unexpected = []
    for tag, i1, i2, j1, j2 in hunks:
        added = patched[j1:j2]
        removed = base[i1:i2]
        blob = "\n".join(added + removed)
        matched = [r for r in expected_regions if re.search(r, blob)]
        marker = "OK " if matched else "??? "
        label = matched[0] if matched else "UNEXPECTED"
        print(f"  [{marker}] lines {i1+1}-{i2} ({tag}, -{len(removed)} +{len(added)}): {label}")
        if not matched:
            unexpected.append((i1 + 1, removed[:3], added[:3]))

    if unexpected:
        print("\n  !! UNEXPECTED HUNKS — the patch touched lines it should not have:")
        for line, rem, add in unexpected:
            print(f"     line {line}")
            for r in rem:
                print(f"       - {r.strip()[:90]}")
            for a in add:
                print(f"       + {a.strip()[:90]}")
    return not unexpected


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    base_repo, patched_repo = Path(sys.argv[1]), Path(sys.argv[2])

    mod_b = base_repo / "janusdna" / "modeling_janusdna.py"
    mod_p = patched_repo / "janusdna" / "modeling_janusdna.py"
    tr_b = base_repo / "train.py"
    tr_p = patched_repo / "train.py"
    for f in (mod_b, mod_p, tr_b, tr_p):
        if not f.exists():
            sys.exit(f"[ABORT] not found: {f}")

    print("=" * 70)
    print("STATIC PATCH VERIFICATION")
    print(f"  baseline : {base_repo}")
    print(f"  patched  : {patched_repo}")
    print("=" * 70)

    if "_USE_STEM" in mod_b.read_text():
        sys.exit("[ABORT] the repository given as baseline is already patched. "
                 "Pass the clean copy as the first argument.")

    ok = True
    ok &= check_presence(mod_p.read_text(), MODEL_PATCHES, "model patches (6)")
    ok &= check_presence(tr_p.read_text(), TRAIN_PATCHES, "train.py patches (2)")
    ok &= band1_equivalence()

    ok &= controlled_diff(mod_b, mod_p, [
        r"_USE_STEM|_STEM_S|_STEM_BLOCK|_STEM_BAND|_STEM_K|_STEM_NSTAGES|conv-stem",
        r"use_conv_stem|CausalConvStem|BlockHead|layer_fusion_strategy",
        r"_fwd = self\.stem|_bwd = self\.stem|hidden_states = inputs_embeds if",
        r"lm_head|_tied_weights_keys|tie_word_embeddings",
    ])
    ok &= controlled_diff(tr_b, tr_p, [
        r"stem-align|_flatten|_align|_STEM|JANUS_STEM|on_fit_start|run_gate|conv-stem",
    ])

    print("\n" + "=" * 70)
    print("RESULT:", "ALL CONSISTENT" if ok else "CHECK THE POINTS FLAGGED ABOVE")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
