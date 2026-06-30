# LatentMAS Quick Statistics

## Language-Level Summary

| lang   |   accuracy |   correct |   total |   latent_reasoning_score |   planner_latent_reasoning_score |   critic_latent_reasoning_score |   refiner_latent_reasoning_score |   judger_latent_reasoning_score |   cosine_to_en |   mean_cosine_to_other |
|:-------|-----------:|----------:|--------:|-------------------------:|---------------------------------:|--------------------------------:|---------------------------------:|--------------------------------:|---------------:|-----------------------:|
| en     |      0.908 |       227 |     250 |                    0.390 |                            0.500 |                           0.500 |                            0.286 |                           0.272 |          1.000 |                  0.762 |
| es     |      0.820 |       205 |     250 |                    0.465 |                            0.431 |                           0.218 |                            0.233 |                           0.976 |          0.813 |                  0.794 |
| de     |      0.812 |       203 |     250 |                    0.462 |                            0.487 |                           0.322 |                            0.288 |                           0.752 |          0.824 |                  0.792 |
| zh     |      0.812 |       203 |     250 |                    0.444 |                            0.504 |                           0.226 |                            0.203 |                           0.844 |          0.764 |                  0.747 |
| ru     |      0.772 |       193 |     250 |                    0.305 |                            0.487 |                           0.219 |                            0.046 |                           0.468 |          0.735 |                  0.748 |
| ja     |      0.748 |       187 |     250 |                    0.537 |                            0.500 |                           0.463 |                            0.305 |                           0.880 |          0.742 |                  0.758 |
| th     |      0.700 |       175 |     250 |                    0.367 |                            0.275 |                           0.205 |                            0.114 |                           0.876 |          0.726 |                  0.754 |
| bn     |      0.476 |       119 |     250 |                    0.318 |                            0.280 |                           0.023 |                            0.000 |                           0.968 |          0.721 |                  0.771 |
| fr     |      0.440 |       110 |     250 |                    0.476 |                            0.448 |                           0.320 |                            0.361 |                           0.776 |          0.816 |                  0.797 |
| sw     |      0.196 |        49 |     250 |                    0.517 |                            0.465 |                           0.367 |                            0.302 |                           0.932 |          0.735 |                  0.745 |
| te     |      0.152 |        38 |     250 |                    0.382 |                            0.433 |                           0.045 |                            0.050 |                           1.000 |          0.742 |                  0.771 |


## Top Shared Latent Reasoning Score Rows

| stage                |   rank_threshold | score                  |   language_pearson |   example_pearson |   score_mean_correct |   score_mean_wrong |
|:---------------------|-----------------:|:-----------------------|-------------------:|------------------:|---------------------:|-------------------:|
| shared_after_planner |              250 | latent_reasoning_score |              0.450 |             0.130 |                0.303 |              0.246 |
| shared_after_planner |                5 | latent_reasoning_score |              0.440 |             0.015 |                0.002 |              0.001 |
| shared_after_planner |               25 | latent_reasoning_score |              0.420 |             0.088 |                0.056 |              0.032 |
| shared_after_critic  |                5 | latent_reasoning_score |              0.405 |             0.025 |                0.006 |              0.003 |
| shared_after_critic  |              250 | latent_reasoning_score |              0.404 |             0.114 |                0.518 |              0.442 |
| shared_after_refiner |              250 | latent_reasoning_score |              0.385 |             0.109 |                0.597 |              0.516 |
| shared_after_refiner |                5 | latent_reasoning_score |              0.383 |             0.032 |                0.009 |              0.005 |
| shared_after_critic  |               25 | latent_reasoning_score |              0.377 |             0.078 |                0.105 |              0.068 |
| shared_with_judger   |              250 | latent_reasoning_score |              0.374 |             0.104 |                0.619 |              0.544 |
| shared_after_planner |               10 | latent_reasoning_score |              0.361 |             0.031 |                0.012 |              0.008 |


## Representative Example

- Language: `en`
- Index: `0`
- Gold: `18`
- Prediction: `18`
- Correct: `True`
- Latent reasoning score: `0.5`

```text
<think>
Let's reason step by step in English.
</think>

</think>

To solve the problem, we need to determine how many eggs Janet sells at the farmers' market each day and then calculate the total amount of money she makes from selling those eggs.

1. **Total eggs laid per day**: Janet’s ducks lay 16 eggs per day.
2. **Eggs eaten for breakfast**: She eats 3 eggs every morning.
3. **Eggs used for muffins**: She uses 4 eggs to bake muffins for her friends every day.
4. **Eggs sold at the market**: Subtract the eggs eaten and used for muffins from the total eggs laid:  
   $ 16 - 3 - 4 = 9 $ eggs.
5. **Money made from selling eggs**: She sells the remaining 9 eggs at $2 per egg:  
   $ 9 \times 2 = 18 $ dollars.

Therefore, Janet makes $\boxed{18}$ dollars every day at the farmers' market.
```
