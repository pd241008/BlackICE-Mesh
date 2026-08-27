# 📜 ADR-004: Memory-Efficient Streaming Data Loader

> **Status:** `Decided`  
> **Date:** August, 2026

---

## 🌎 Context

The multi-seed training pipeline (seeds 42–54, 3 methods, 3 datasets) required running multiple sequential training jobs on a machine with ~11GB RAM. Initial attempts at concurrent multi-seed training crashed with OS-level `Killed` signals — OOM kills from the kernel — because each training process attempted to load entire Parquet datasets into memory before beginning training.

Dataset sizes at the time:
- `cicids2017_train.parquet`: ~2.5M rows, ~78 features → ~1.5 GB in memory
- `unsw_nb15_train.parquet`: ~2.5M rows, ~79 features → ~1.5 GB in memory
- `nsl-kdd` (CSV): ~125k rows — small enough to be non-issue

With three concurrent processes, total memory demand exceeded 9 GB for data alone, before accounting for model weights, optimizer state, and gradient buffers.

## 🛤️ Options Considered

1. **Reduce batch size globally** — Reduces peak tensor size but does not reduce the memory cost of loading the full dataset upfront.
2. **Subsample training data to NSL-KDD size (125k rows)** — Already done for the first ablation round. Throwing away 95% of the data is valid for controlled ablations but not for the final multi-seed runs where we want full dataset statistics.
3. **Streaming Parquet loader (IterableDataset)** — Stream chunks from the Parquet file directly into the training loop using `pyarrow.parquet.ParquetFile.iter_batches()`. Memory footprint becomes O(batch_size) rather than O(dataset_size).
4. **Sequential-only execution** — Enforce one job at a time via shell script, eliminating concurrency as the source of OOM even with the full in-memory loader.

---

## 🎯 Decision

> [!IMPORTANT]  
> **Implement Option 3 (streaming) and Option 4 (sequential execution) together.**  
> Streaming is the primary fix; sequential execution is the safety net that guarantees no regression if a future caller inadvertently runs jobs in parallel.

## 🧠 Reasoning

`StreamingParquetDataset` wraps PyArrow's `iter_batches()` in a `torch.utils.data.IterableDataset`. Each batch is yielded, consumed, and garbage-collected before the next is read. Peak memory for a 32,768-row batch of 196 features is approximately 24 MB — a 60x reduction compared to loading the full UNSW-NB15 dataset.

The sequential shell script (`resume_multiseed_sequential.sh`) additionally provides:
- **Resume-on-restart:** A skip-if-model-exists check means server restarts (which occurred 3 times during this session) do not require retraining completed models.
- **Predictable resource profile:** One Python process at a time; memory never exceeds ~1.5 GB regardless of dataset.

**Batch size was set to 32,768** (from an earlier conservative 4,096) once sequential execution was confirmed stable. This gave a ~8x throughput improvement with no memory regression.

**Connection to evaluation fidelity:** The streaming infrastructure validated here is what enables faithful full-scale evaluation at n=22,543 (vs. the small n=100 that overestimates robust accuracy by 7.64pp). The exhaustive evaluator also scales near-linearly: 14→42 candidates at 1.4→7.8s on UNSW-NB15, with K=2 (667 candidates) completing in 59.0s on an RTX 4050 Laptop.

_See `app/ml/data/loader.py` → `StreamingParquetDataset` class._  
_See `scratch/resume_multiseed_sequential.sh`._
