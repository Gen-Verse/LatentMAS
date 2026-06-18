# L-R Disentanglement Steering Figures

These figures were produced from existing LatentMAS CSV outputs. No LRS or hidden-state recomputation is used.

## Included Runs

- `mgsm_first50_latent_mas_baseline`: strength `0`, `final`, source `src/multilingual-latent-reasoning/results_latent_mas_agents/Qwen3-4B/mgsm_first50_latent_mas_baseline/latent_agent_similarity_examples.csv`
- `mgsm_first50_latent_mas_lr_disentangle_s0.025`: strength `0.025`, `final`, source `src/multilingual-latent-reasoning/results_latent_mas_agents/Qwen3-4B/mgsm_first50_latent_mas_lr_disentangle_s0.025/latent_agent_similarity_examples.csv`
- `mgsm_first50_latent_mas_lr_disentangle_s0.05`: strength `0.05`, `final`, source `src/multilingual-latent-reasoning/results_latent_mas_agents/Qwen3-4B/mgsm_first50_latent_mas_lr_disentangle_s0.05/latent_agent_similarity_examples.csv`
- `mgsm_first50_latent_mas_lr_disentangle_s0.1`: strength `0.1`, `partial`, source `src/multilingual-latent-reasoning/results_latent_mas_agents/Qwen3-4B/mgsm_first50_latent_mas_lr_disentangle_s0.1/latent_agent_similarity_examples.partial.csv`

## Figures

### Baseline Vs Best Steered Accuracy

![Baseline Vs Best Steered Accuracy](baseline_vs_best_steered_accuracy.png)

### Delta Accuracy Heatmap Vs Baseline

![Delta Accuracy Heatmap Vs Baseline](delta_accuracy_heatmap_vs_baseline.png)

### Language Accuracy Curves

![Language Accuracy Curves](language_accuracy_curves.png)

### Macro Accuracy Vs Steering Strength

![Macro Accuracy Vs Steering Strength](macro_accuracy_vs_steering_strength.png)

### Net Fixed Problem Heatmap Vs Baseline

![Net Fixed Problem Heatmap Vs Baseline](net_fixed_problem_heatmap_vs_baseline.png)

### Problem Indices Most Helped Hurt

![Problem Indices Most Helped Hurt](problem_indices_most_helped_hurt.png)
