"""Downstream evaluation of hybrid models (early Nesterov/TMM + late Vanilla).

Usage:
  python hybrid_eval_run.py --base-model yurii --cutoff-layer 6 \
      --checkpoint path/to/best.pt --output-dir eval_results_hybrid_yurii6
"""

import argparse
import json
import os

import numpy as np
import lm_eval

from hybrid_eval_model import HybridFormerLM


def main():
    parser = argparse.ArgumentParser(description="Evaluate hybrid model on downstream tasks")
    parser.add_argument("--base-model", required=True, choices=["yurii", "tmm"])
    parser.add_argument("--cutoff-layer", type=int, default=6,
                        help="Layers 0..cutoff-1 use Nesterov/TMM; cutoff..11 use Vanilla")
    parser.add_argument("--nesterov-layers", type=str, default=None,
                        help='Comma-separated layer indices that keep Nesterov/TMM dynamics '
                             '(e.g. "0,1,11"). Overrides --cutoff-layer when set. '
                             'All other layers run Vanilla residual; velocity is passed '
                             'through Vanilla layers unchanged.')
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]

    nesterov_layers = None
    if args.nesterov_layers:
        nesterov_layers = [int(s) for s in args.nesterov_layers.split(",") if s.strip()]

    print(f"Loading {args.base_model} from {args.checkpoint}")
    if nesterov_layers is not None:
        print(f"Hybrid layer set: layers {sorted(nesterov_layers)} = {args.base_model}, "
              f"all others = vanilla")
    else:
        print(f"Hybrid cutoff: layers 0-{args.cutoff_layer - 1} = {args.base_model}, "
              f"layers {args.cutoff_layer}-11 = vanilla")
    lm = HybridFormerLM(
        checkpoint_path=args.checkpoint,
        base_model=args.base_model,
        cutoff_layer=args.cutoff_layer,
        nesterov_layers=nesterov_layers,
        device=args.device,
    )

    tasks = [
        ("hellaswag", 10),
        ("arc_easy", 25),
    ]

    for task_name, num_fewshot in tasks:
        print(f"\n{'='*60}")
        print(f"Evaluating {task_name} ({num_fewshot}-shot)...")
        print(f"{'='*60}")

        results = lm_eval.simple_evaluate(
            model=lm,
            tasks=[task_name],
            num_fewshot=num_fewshot,
            log_samples=True,
            limit=args.limit,
            random_seed=0,
            numpy_random_seed=1234,
            torch_random_seed=1234,
            fewshot_random_seed=1234,
        )

        for sample in results.get("samples", {}).get(task_name, []):
            resps = sample["filtered_resps"]
            lls = np.array([r[0] for r in resps])
            choices = sample["doc"]["choices"]
            if isinstance(choices, dict):
                choices = choices["text"]
            cont_lens = np.array([float(len(c)) for c in choices])
            sample["pred"] = int(np.argmax(lls))
            sample["pred_norm"] = int(np.argmax(lls / cont_lens))

        output_path = os.path.join(args.output_dir, f"{task_name}_{ckpt_name}.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {output_path}")

        task_results = results["results"].get(task_name, {})
        acc_norm = task_results.get("acc_norm,none", task_results.get("acc_norm", "N/A"))
        acc = task_results.get("acc,none", task_results.get("acc", "N/A"))
        print(f"  acc_norm = {acc_norm}")
        print(f"  acc      = {acc}")

    print(f"\n{'='*60}")
    print("Evaluation complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
