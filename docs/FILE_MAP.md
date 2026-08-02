# File map

The scripts in this repository were written during the thesis work with
numbered, Italian-language names. They were renamed for publication. This table
is the correspondence, for anyone cross-referencing the thesis text or the
original working directory.

## Downstream pipeline (`src/`)

| Original | Current |
|---|---|
| `common.py` | `src/common.py` |
| `01_data_prep.py` | `src/prepare_windows.py` |
| `03_extract_crossdna.py` | `src/extract_crossdna.py` |
| `03_extract_janusdna.py` | `src/extract_janusdna.py` |
| `03_extract_janusdna_stem.py` | merged into `src/extract_janusdna.py` (automatic) |
| `04_cluster.py` | `src/cluster_embeddings.py` |
| `05_classify.py` | `src/classify_baselines.py` |
| `05b_trivial_decompose.py` | `src/decompose_trivial.py` |
| `07_mlp_probe.py` | `src/mlp_probe.py` |
| `08_mlp_probe_compare.py` | `src/mlp_probe_compare.py` |
| `06b_finetune_lora.py` | `src/finetune_lora_crossdna.py` |
| `06b_finetune_lora_janus.py` | `src/finetune_lora_janusdna.py` |
| `06b_finetune_lora_janus_stem.py` | merged into `src/finetune_lora_janusdna.py` (`--stem`) |
| `06c_eval_lora.py` | `src/eval_lora.py` |

## Pre-training (`pretrain/`)

| Original | Current |
|---|---|
| `make_pretrain_bed.py` | `pretrain/make_pretrain_bed.py` |
| `DEF/scripts/make_pretrain_bed_clean.py` | `pretrain/make_pretrain_bed_clean.py` |
| `make_penguin_bed.py` | `pretrain/make_downstream_bed.py` |
| `pretrain_janusdna_penguin_1gpu.sh` | `pretrain/pretrain_janusdna.sh` |
| `scripts/pretrain_crossdna_penguin_1gpu.sh` | `pretrain/pretrain_crossdna.sh` |
| `convert_crossdna_ckpt_to_hf.py` | `pretrain/crossdna_ckpt_to_hf.py` |
| `run_janus_clean_eval.sh` | `pretrain/eval_after_pretrain.sh` |

## Conv-stem experiment (`stem/`)

| Original | Current |
|---|---|
| `causal_conv_stem.py` | `stem/causal_conv_stem.py` |
| `test_stem_wiring.py` | `stem/test_stem_wiring.py` |
| `run_stem_experiment.sh` | `stem/run_stem_experiment.sh` |
| `extract_stem.sh` | `stem/extract_stem.sh` |
| `finetune_lora_stem.sh` | `stem/finetune_lora_stem.sh` |
| `JanusDNA_stem*/` (4 full repo copies) | `stem/janusdna_stem.patch` |

## Analysis (`analysis/`)

| Original | Current |
|---|---|
| `make_figures_DEF.py` | `analysis/make_figures.py` |
| `make_figure_sweep_L.py` | `analysis/make_figure_window_sweep.py` |
| `analyze_embeddings.py` | `analysis/analyze_embeddings.py` |
| `09_confusion_compare.py` | `analysis/compare_confusion.py` |
| `confronta_confusioni.py` | `analysis/recall_vs_window.py` |
| `per_class_accuracy.py` | `analysis/per_class_accuracy.py` |
| — (new) | `analysis/audit_results.py` |

## Tools (`tools/`)

| Original | Current |
|---|---|
| `02_smoke.py` | `tools/smoke_test_crossdna.py` |
| `check_embeddings_sanity.py` | `tools/check_embeddings.py` |
| `verify_lora_targets.py` | `tools/verify_lora_targets.py` |
| `certify_patch.py` | `tools/certify_fp32_patch.py` |

## Environment (`env/`)

| Original | Current |
|---|---|
| `crossdna_noflash.yml` | `env/crossdna.yml` |
| `crossdna_base.yml` | `env/crossdna_base.yml` |
| `startup.sh` + `scripts/startup.sh` | `env/container_setup.sh` (merged) |
| `sdpa_patch.py` | `env/patches/sdpa_fp32.py` |
| `flexattn_blocksize_patch.py` | `env/patches/flexattn_blocksize.py` |

## Not published

| Original | Reason |
|---|---|
| `00_inspect_config.py` | 7-line debug helper |
| `tokenizer_test.py` | 19-line scratch check |
| `02_smoke.py` variants | superseded |
| `06_finetune.py` | full fine-tuning, abandoned in favour of LoRA |
| `06d_finetune_lora_resume.py` | one-off recovery after a container was destroyed |
| `10_make_figures.py` | superseded by `make_figures_DEF.py` |
| `run_extraction_janusdna.sh`, `run_extraction_janusdna72.sh` | superseded by `eval_after_pretrain.sh` |
| `cleanup_artifacts.sh`, `collect_for_github.sh` | workspace housekeeping, not part of the pipeline |
| `DEF/scripts/*` | snapshot duplicates (8 of 11 byte-identical to the working copies) |
| `consolidate_DEF.py` | workspace-specific archival script: copies ~40 artifacts between absolute paths that exist only on the original server. Its verification role — reading real numbers from the JSONs rather than transcribing them — is covered generically by `analysis/audit_results.py`. |
