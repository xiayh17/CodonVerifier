#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple


class ReservoirSampler:
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.samples: List[float] = []
        self._n_seen = 0

    def add(self, value: float) -> None:
        self._n_seen += 1
        if len(self.samples) < self.capacity:
            self.samples.append(value)
            return
        # Reservoir sampling replacement probability
        # Use a simple modular skip to avoid importing random for speed
        # If desired, replace with random for strict uniformity
        idx = self._n_seen % self.capacity
        self.samples[idx] = value

    def quantiles(self, probs: List[float]) -> Dict[str, Optional[float]]:
        if not self.samples:
            return {f"q{int(p*100)}": None for p in probs}
        s = sorted(self.samples)
        n = len(s)
        out: Dict[str, Optional[float]] = {}
        for p in probs:
            # nearest-rank
            k = max(0, min(n - 1, int(round(p * (n - 1)))))
            out[f"q{int(p*100)}"] = float(s[k])
        return out


class NumericStats:
    def __init__(self, quantile_capacity: int = 10000):
        self.count = 0
        self.missing = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.min_val: Optional[float] = None
        self.max_val: Optional[float] = None
        self.sampler = ReservoirSampler(quantile_capacity)

    def add(self, value: Optional[float]) -> None:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            self.missing += 1
            return
        try:
            x = float(value)
        except Exception:
            self.missing += 1
            return
        self.count += 1
        if self.min_val is None or x < self.min_val:
            self.min_val = x
        if self.max_val is None or x > self.max_val:
            self.max_val = x
        # Welford online algorithm
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.m2 += delta * delta2
        self.sampler.add(x)

    def finalize(self) -> Dict[str, Any]:
        variance = (self.m2 / (self.count - 1)) if self.count > 1 else 0.0
        std = math.sqrt(variance) if variance >= 0 else float("nan")
        quantiles = self.sampler.quantiles([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        return {
            "count": self.count,
            "missing": self.missing,
            "mean": self.mean if self.count > 0 else None,
            "std": std if self.count > 1 else None,
            "min": self.min_val,
            "p01": quantiles.get("q1"),
            "p05": quantiles.get("q5"),
            "p25": quantiles.get("q25"),
            "p50": quantiles.get("q50"),
            "p75": quantiles.get("q75"),
            "p95": quantiles.get("q95"),
            "p99": quantiles.get("q99"),
            "max": self.max_val,
        }


class CategoricalStats:
    def __init__(self):
        self.count = 0
        self.missing = 0
        self.counter: Counter = Counter()

    def add(self, value: Any) -> None:
        if value is None:
            self.missing += 1
            return
        self.count += 1
        self.counter[str(value)] += 1

    def finalize(self, top_k: int = 50) -> Dict[str, Any]:
        most_common = self.counter.most_common(top_k)
        return {
            "count": self.count,
            "missing": self.missing,
            "unique": len(self.counter),
            "top_k": most_common,
        }


class LengthStats:
    def __init__(self):
        self.num_items = 0
        self.missing = 0
        self.length_numeric = NumericStats()
        self.l2_numeric = NumericStats()

    def add(self, seq: Optional[Iterable[Any]]) -> None:
        if seq is None:
            self.missing += 1
            return
        try:
            lst = list(seq)
        except Exception:
            self.missing += 1
            return
        self.num_items += 1
        self.length_numeric.add(len(lst))
        # If elements are numeric, compute L2 norm
        is_numeric = True
        total_sq = 0.0
        for v in lst:
            try:
                fv = float(v)
                if math.isnan(fv):
                    is_numeric = False
                    break
                total_sq += fv * fv
            except Exception:
                is_numeric = False
                break
        if is_numeric:
            self.l2_numeric.add(math.sqrt(total_sq))

    def finalize(self) -> Dict[str, Any]:
        out = {
            "items": self.num_items,
            "missing": self.missing,
            "length": self.length_numeric.finalize(),
        }
        l2 = self.l2_numeric.finalize()
        if l2["count"] > 0:
            out["l2_norm"] = l2
        return out


def is_number(value: Any) -> bool:
    try:
        if value is None:
            return False
        float(value)
        return True
    except Exception:
        return False


def infer_type_from_sample(values: List[Any]) -> str:
    # Returns one of: "numeric", "categorical", "list", "dict", "other"
    for v in values:
        if v is None:
            continue
        if isinstance(v, (int, float)):
            return "numeric"
        if isinstance(v, (list, tuple)):
            return "list"
        if isinstance(v, dict):
            return "dict"
        if isinstance(v, (str, bool)):
            return "categorical"
    return "other"


class StreamingCorrelation:
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.k = len(feature_names)
        self.n = 0
        self.mean = [0.0] * self.k
        # co-moment matrix upper triangle stored fully (k x k)
        self.comoment = [[0.0 for _ in range(self.k)] for _ in range(self.k)]

    def add(self, vector: List[Optional[float]]) -> None:
        # Require full observation to update correlation
        if any(v is None for v in vector):
            return
        x = [float(v) for v in vector]
        self.n += 1
        if self.n == 1:
            self.mean = x[:]  # initialize mean
            return
        # Chan et al. incremental covariance update
        delta = [x[i] - self.mean[i] for i in range(self.k)]
        r = self.n
        for i in range(self.k):
            self.mean[i] += delta[i] / r
        delta2 = [x[i] - self.mean[i] for i in range(self.k)]
        for i in range(self.k):
            di = delta[i]
            d2i = delta2[i]
            for j in range(i, self.k):
                self.comoment[i][j] += di * delta2[j]

    def finalize(self) -> Tuple[List[List[float]], List[float]]:
        if self.n < 2:
            zero_mat = [[0.0 for _ in range(self.k)] for _ in range(self.k)]
            return zero_mat, [0.0] * self.k
        variances = [self.comoment[i][i] / (self.n - 1) for i in range(self.k)]
        stds = [math.sqrt(v) if v > 0 else 0.0 for v in variances]
        corr = [[1.0 if i == j else 0.0 for j in range(self.k)] for i in range(self.k)]
        for i in range(self.k):
            for j in range(i + 1, self.k):
                denom = stds[i] * stds[j]
                if denom == 0:
                    rho = 0.0
                else:
                    cov_ij = self.comoment[i][j] / (self.n - 1)
                    rho = max(-1.0, min(1.0, cov_ij / denom))
                corr[i][j] = rho
                corr[j][i] = rho
        return corr, stds


def sanitize_feature_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", name)


def process_file(path: str, out_dir: str, max_rows: int = 0, type_infer_rows: int = 200,
                 max_categorical_topk: int = 100, max_dict_keys: int = 200) -> Dict[str, Any]:
    species = os.path.basename(path).split("_")[0]
    os.makedirs(out_dir, exist_ok=True)

    # First pass: infer field types from a small sample
    field_samples: Dict[str, List[Any]] = defaultdict(list)
    total_rows = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            for k, v in obj.items():
                if len(field_samples[k]) < type_infer_rows:
                    field_samples[k].append(v)
            total_rows += 1
            if type_infer_rows and total_rows >= type_infer_rows:
                break

    field_types: Dict[str, str] = {}
    for k, vals in field_samples.items():
        field_types[k] = infer_type_from_sample(vals)

    # Prepare stats collectors
    numeric_scalars: Dict[str, NumericStats] = {}
    categorical_scalars: Dict[str, CategoricalStats] = {}
    list_stats: Dict[str, LengthStats] = {}
    dict_numeric_stats: Dict[str, Dict[str, NumericStats]] = {}
    dict_key_counts: Dict[str, Counter] = {}

    # Correlation on numeric scalars only
    numeric_scalar_names: List[str] = [k for k, t in field_types.items() if t == "numeric"]
    numeric_scalar_names.sort()
    corr_tracker = StreamingCorrelation(numeric_scalar_names) if numeric_scalar_names else None

    # Second pass: full stream
    n_streamed = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            # Scalars
            for name in numeric_scalar_names:
                if name not in numeric_scalars:
                    numeric_scalars[name] = NumericStats()
                numeric_scalars[name].add(obj.get(name))
            for name, t in field_types.items():
                v = obj.get(name)
                if t == "categorical":
                    if name not in categorical_scalars:
                        categorical_scalars[name] = CategoricalStats()
                    categorical_scalars[name].add(v)
                elif t == "list":
                    if name not in list_stats:
                        list_stats[name] = LengthStats()
                    list_stats[name].add(v)
                elif t == "dict":
                    if isinstance(v, dict):
                        if name not in dict_numeric_stats:
                            dict_numeric_stats[name] = {}
                            dict_key_counts[name] = Counter()
                        for dk, dv in v.items():
                            dict_key_counts[name][dk] += 1
                            if is_number(dv):
                                if len(dict_numeric_stats[name]) < max_dict_keys or dk in dict_numeric_stats[name]:
                                    if dk not in dict_numeric_stats[name]:
                                        dict_numeric_stats[name][dk] = NumericStats()
                                    dict_numeric_stats[name][dk].add(float(dv))
                # numeric handled above; other types ignored

            # Correlation sample vector
            if corr_tracker is not None:
                vec = [obj.get(n) if is_number(obj.get(n)) else None for n in numeric_scalar_names]
                corr_tracker.add(vec)

            n_streamed += 1
            if max_rows and n_streamed >= max_rows:
                break

    # Finalize stats
    numeric_summary_rows: List[Dict[str, Any]] = []
    for name in numeric_scalar_names:
        stats = numeric_scalars[name].finalize()
        numeric_summary_rows.append({"feature": name, **stats})

    # Write numeric CSV
    csv_path = os.path.join(out_dir, f"{species}_numeric_scalars.csv")
    if numeric_summary_rows:
        fieldnames = list(numeric_summary_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as wf:
            writer = csv.DictWriter(wf, fieldnames=fieldnames)
            writer.writeheader()
            for row in numeric_summary_rows:
                writer.writerow(row)

    # Correlation CSV
    if corr_tracker is not None and numeric_scalar_names:
        corr_matrix, stds = corr_tracker.finalize()
        corr_csv = os.path.join(out_dir, f"{species}_numeric_correlation.csv")
        with open(corr_csv, "w", newline="", encoding="utf-8") as wf:
            writer = csv.writer(wf)
            writer.writerow(["feature"] + numeric_scalar_names)
            for i, name in enumerate(numeric_scalar_names):
                writer.writerow([name] + [f"{corr_matrix[i][j]:.6f}" for j in range(len(numeric_scalar_names))])

    # JSON summary (categorical, lists, dicts)
    json_summary: Dict[str, Any] = {
        "file": path,
        "species": species,
        "rows_processed": n_streamed,
        "numeric_scalar_features": len(numeric_scalar_names),
        "categorical_scalar_features": len([1 for t in field_types.values() if t == "categorical"]),
        "list_features": len([1 for t in field_types.values() if t == "list"]),
        "dict_features": len([1 for t in field_types.values() if t == "dict"]),
        "categorical": {},
        "lists": {},
        "dict_numeric": {},
    }

    # Categorical details
    for name, cat in categorical_scalars.items():
        json_summary["categorical"][name] = cat.finalize(top_k=max_categorical_topk)

    # List details
    for name, lst in list_stats.items():
        json_summary["lists"][name] = lst.finalize()

    # Dict numeric details
    for fname, key_to_stats in dict_numeric_stats.items():
        # Report top keys by presence
        key_presence = dict_key_counts.get(fname, Counter())
        top_keys = [k for k, _ in key_presence.most_common(max_dict_keys)]
        per_key = {}
        for k in top_keys:
            if k in key_to_stats:
                per_key[k] = key_to_stats[k].finalize()
        json_summary["dict_numeric"][fname] = {
            "top_keys": top_keys,
            "stats": per_key,
        }

    json_path = os.path.join(out_dir, f"{species}_summary.json")
    with open(json_path, "w", encoding="utf-8") as wf:
        json.dump(json_summary, wf, ensure_ascii=False, indent=2)

    return {
        "species": species,
        "rows": n_streamed,
        "numeric_csv": csv_path if numeric_summary_rows else None,
        "corr_csv": os.path.join(out_dir, f"{species}_numeric_correlation.csv") if numeric_scalar_names else None,
        "json_summary": json_path,
    }


def find_input_files(patterns: List[str]) -> List[str]:
    files: List[str] = []
    for p in patterns:
        if os.path.isdir(p):
            for root, _, fnames in os.walk(p):
                for fn in fnames:
                    if fn.endswith("_complete_v2.jsonl"):
                        files.append(os.path.join(root, fn))
        elif os.path.isfile(p):
            files.append(p)
        else:
            # Glob-like simple support: directory segment + suffix
            base_dir = os.path.dirname(p) or "."
            suffix = os.path.basename(p)
            try:
                for fn in os.listdir(base_dir):
                    if fn.endswith(suffix):
                        files.append(os.path.join(base_dir, fn))
            except Exception:
                pass
    # Deduplicate and sort for determinism
    files = sorted(list(dict.fromkeys(files)))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute descriptive statistics from JSONL feature files")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input files or directories or simple suffix globs")
    parser.add_argument("--out", required=True, help="Output directory for reports")
    parser.add_argument("--max-rows", type=int, default=0, help="Maximum rows to process per file (0=all)")
    parser.add_argument("--infer-rows", type=int, default=200, help="Rows to sample for type inference")
    parser.add_argument("--topk", type=int, default=100, help="Top-K categories to report")
    parser.add_argument("--max-dict-keys", type=int, default=200, help="Max dict keys to summarize per dict feature")
    args = parser.parse_args()

    inputs = find_input_files(args.inputs)
    if not inputs:
        print("No input files found", file=sys.stderr)
        sys.exit(2)

    os.makedirs(args.out, exist_ok=True)

    index_rows: List[Dict[str, str]] = []
    for path in inputs:
        try:
            result = process_file(
                path=path,
                out_dir=args.out,
                max_rows=args.max_rows,
                type_infer_rows=args.infer_rows,
                max_categorical_topk=args.topk,
                max_dict_keys=args.max_dict_keys,
            )
        except Exception as e:
            print(f"Error processing {path}: {e}", file=sys.stderr)
            continue
        species = result.get("species", os.path.basename(path))
        json_summary = result.get("json_summary") or ""
        numeric_csv = result.get("numeric_csv") or ""
        corr_csv = result.get("corr_csv") or ""
        index_rows.append({
            "species": species,
            "file": path,
            "json_summary": json_summary,
            "numeric_csv": numeric_csv,
            "corr_csv": corr_csv,
        })

    # Write an index CSV for convenience
    index_csv = os.path.join(args.out, "index.csv")
    with open(index_csv, "w", newline="", encoding="utf-8") as wf:
        writer = csv.DictWriter(wf, fieldnames=["species", "file", "json_summary", "numeric_csv", "corr_csv"])
        writer.writeheader()
        for row in index_rows:
            writer.writerow(row)

    print(f"Wrote reports to: {args.out}")
    for row in index_rows:
        print(f" - {row['species']}: summary={row['json_summary']} numeric={row['numeric_csv']} corr={row['corr_csv']}")


if __name__ == "__main__":
    main()


