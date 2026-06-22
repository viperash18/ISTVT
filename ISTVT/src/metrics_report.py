"""
metrics_report.py
=================
Compute a FULL evaluation report (confusion matrix, precision / recall / F1,
specificity, balanced accuracy, AUROC) from the per-video scores that
evaluate.py already saved. It reads only the results JSON -- no model, no GPU,
no torch -- so it runs anywhere (the free CPU Studio is fine).

    python -m src.metrics_report
    python -m src.metrics_report --results logs/<run>_test_results.json
    python -m src.metrics_report --threshold 0.5

It prints the metrics and, if matplotlib is available, also saves an ROC curve
and a confusion-matrix image under visualizations/.
"""

import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg


def _roc_auc(labels, probs):
    """AUROC via Mann-Whitney U (pure NumPy, handles ties)."""
    labels = np.asarray(labels)
    probs = np.asarray(probs, dtype=float)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(probs, kind="mergesort")
    sp = probs[order]
    ranks = np.empty(len(probs), dtype=float)
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _roc_curve(labels, probs, n=200):
    ths = np.linspace(0.0, 1.0, n)
    P = max(1, int((labels == 1).sum()))
    N = max(1, int((labels == 0).sum()))
    fpr, tpr = [], []
    for t in ths:
        pred = (probs >= t).astype(int)
        tpr.append(int(((pred == 1) & (labels == 1)).sum()) / P)
        fpr.append(int(((pred == 1) & (labels == 0)).sum()) / N)
    return np.array(fpr), np.array(tpr)


def best_threshold(labels, probs):
    best_t, best_acc = 0.5, 0.0
    for t in np.linspace(0.01, 0.99, 99):
        acc = float(((probs >= t).astype(int) == labels).mean())
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t, best_acc


def _confusion(labels, probs, t):
    pred = (probs >= t).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    return tp, tn, fp, fn


def _report(labels, probs, t, title):
    tp, tn, fp, fn = _confusion(labels, probs, t)
    n = tp + tn + fp + fn
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0       # of predicted-fake, how many are fake
    rec = tp / (tp + fn) if (tp + fn) else 0.0        # recall / sensitivity (catch fakes)
    spec = tn / (tn + fp) if (tn + fp) else 0.0       # specificity (correct on reals)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    bal = (rec + spec) / 2.0
    print(f"\n--- {title}  (threshold = {t:.2f}) ---")
    print("Confusion matrix (rows = actual, cols = predicted):")
    print("                  pred REAL   pred FAKE")
    print(f"   actual REAL      {tn:7d}    {fp:7d}")
    print(f"   actual FAKE      {fn:7d}    {tp:7d}")
    print(f"   accuracy            : {acc * 100:6.2f}%")
    print(f"   balanced accuracy   : {bal * 100:6.2f}%")
    print(f"   precision (fake)    : {prec * 100:6.2f}%")
    print(f"   recall / sensitivity: {rec * 100:6.2f}%")
    print(f"   specificity (real)  : {spec * 100:6.2f}%")
    print(f"   F1 score            : {f1 * 100:6.2f}%")
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "accuracy": acc,
            "balanced_accuracy": bal, "precision": prec, "recall": rec,
            "specificity": spec, "f1": f1, "threshold": t}


def main():
    ap = argparse.ArgumentParser()
    default_results = os.path.join(cfg.LOG_DIR, cfg.RUN_NAME + "_test_results.json")
    ap.add_argument("--results", default=default_results,
                    help="results JSON saved by evaluate.py")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    if not os.path.isfile(args.results):
        raise FileNotFoundError(
            f"Results file not found: {args.results}\n"
            f"Run `python -m src.evaluate` first to produce it.")

    with open(args.results) as f:
        data = json.load(f)
    vs = data["video_scores"]
    vl = data["video_labels"]
    vids = list(vs.keys())
    probs = np.array([float(vs[v]) for v in vids], dtype=float)
    labels = np.array([int(vl[v]) for v in vids])

    print("=" * 60)
    print(f"EVALUATION REPORT  --  {os.path.basename(args.results)}")
    print(f"videos = {len(vids)}   real = {int((labels == 0).sum())}   "
          f"fake = {int((labels == 1).sum())}")
    print("=" * 60)
    print(f"AUROC (threshold-free) : {_roc_auc(labels, probs) * 100:.2f}%")

    summary = {"auc": _roc_auc(labels, probs)}
    summary["at_fixed"] = _report(labels, probs, args.threshold, "At fixed threshold")
    bt, _ = best_threshold(labels, probs)
    summary["at_best"] = _report(labels, probs, bt, "At best-accuracy threshold")

    # save the numeric summary next to the input
    out_json = args.results.replace(".json", "_metrics.json")
    json.dump(summary, open(out_json, "w"), indent=2)
    print(f"\nsaved metrics summary -> {out_json}")

    # optional plots -- guarded so the text report above always works
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(cfg.VIS_DIR, exist_ok=True)
        fpr, tpr = _roc_curve(labels, probs)
        auc = _roc_auc(labels, probs)
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, label=f"AUROC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC -- cross-dataset ({cfg.CROSS_TEST_DATASET})")
        plt.legend(loc="lower right")
        plt.tight_layout()
        roc_path = os.path.join(cfg.VIS_DIR, cfg.RUN_NAME + "_roc.png")
        plt.savefig(roc_path, dpi=150)
        plt.close()

        tp, tn, fp, fn = _confusion(labels, probs, args.threshold)
        cm = np.array([[tn, fp], [fn, tp]])
        plt.figure(figsize=(4.2, 4))
        plt.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            plt.text(j, i, str(v), ha="center", va="center", fontsize=14,
                     color="white" if v > cm.max() / 2 else "black")
        plt.xticks([0, 1], ["pred REAL", "pred FAKE"])
        plt.yticks([0, 1], ["REAL", "FAKE"])
        plt.title(f"Confusion matrix (t = {args.threshold:.2f})")
        plt.tight_layout()
        cm_path = os.path.join(cfg.VIS_DIR, cfg.RUN_NAME + "_confusion.png")
        plt.savefig(cm_path, dpi=150)
        plt.close()
        print(f"saved plots:\n  {roc_path}\n  {cm_path}")
    except Exception as e:                              # noqa: BLE001
        print(f"\n(plots skipped: {e})")
        print("The text metrics above are complete; plotting just needs matplotlib.")


if __name__ == "__main__":
    main()
