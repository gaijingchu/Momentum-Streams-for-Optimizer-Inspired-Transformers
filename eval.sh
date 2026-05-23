mkdir -p logs eval_results

uv run python eval_run.py --checkpoint checkpoints/best.pt --output-dir eval_results
