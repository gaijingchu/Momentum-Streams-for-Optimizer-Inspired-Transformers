"""Downstream evaluation of AdamWFormer using lm-evaluation-harness v0.4.3."""

import argparse
import json
import os

import numpy as np
import lm_eval

from adamw_eval_model import AdamWFormerLM


def main():
    parser = argparse.ArgumentParser(description="Evaluate AdamWFormer on downstream tasks")
    parser.add_argument("--checkpoint", default="checkpoints_adamw/best.pt", help="Path to checkpoint")
    parser.add_argument("--output-dir", default="eval_results_adamw", help="Output directory")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--limit", type=int, default=None, help="Limit examples per task (for debugging)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]

    print(f"Loading model from {args.checkpoint}...")
    lm = AdamWFormerLM(checkpoint_path=args.checkpoint, device=args.device)

    # Task configs: (task_name, num_fewshot)
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

        # Add predicted choice indices to each sample
        for sample in results.get("samples", {}).get(task_name, []):
            resps = sample["filtered_resps"]
            lls = np.array([r[0] for r in resps])
            choices = sample["doc"]["choices"]
            if isinstance(choices, dict):
                choices = choices["text"]
            cont_lens = np.array([float(len(c)) for c in choices])
            sample["pred"] = int(np.argmax(lls))
            sample["pred_norm"] = int(np.argmax(lls / cont_lens))

        # Save results
        output_path = os.path.join(args.output_dir, f"{task_name}_{ckpt_name}.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {output_path}")

        # Print metrics
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
