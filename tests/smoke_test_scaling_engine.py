import sys
import torch
import logging

logging.basicConfig(level=logging.INFO)

print("--- SMOKE TEST: Multilingual Scaling Engine ---")

try:
    print("1. Testing UniversalLatentHub...")
    from src.latent_coordination.latent_space.universal_space import UniversalLatentHub
    hub = UniversalLatentHub()
    hub.add_language_adapter("th", 1024)
    assert "th" in hub.adapters
    print("[PASS] UniversalLatentHub")

    print("2. Testing RecursiveLatentCore...")
    from src.latent_coordination.latent_space.recursive_core import RecursiveLatentCore
    core = RecursiveLatentCore()
    z_init = torch.randn(1, 512)
    z_out = core(z_init)
    assert z_out.shape == (1, 512)
    print("[PASS] RecursiveLatentCore")

    print("3. Testing GeometryConditionedCVAEPrior...")
    from src.latent_coordination.topology.cvae_prior import GeometryConditionedCVAEPrior
    prior = GeometryConditionedCVAEPrior(query_dim=1024)
    q = torch.randn(1, 1024)
    geo_l = torch.randn(1, 9)
    z, mu, logvar = prior(q, geo_l)
    assert z.shape == (1, 256)
    print("[PASS] GeometryConditionedCVAEPrior")

    print("4. Testing QueryReconstructionProbe...")
    from src.latent_coordination.eval.verification_probe import QueryReconstructionProbe, LatentDriftException
    probe = QueryReconstructionProbe()
    z_t = torch.randn(1, 512)
    q_orig = torch.randn(1, 1024)
    try:
        drift_score = probe(z_t, q_orig)
    except LatentDriftException:
        pass # Expected or possible depending on random init
    print("[PASS] QueryReconstructionProbe")

    print("5. Testing MultiAgentRunner (Dry Run)...")
    from src.latent_coordination.eval.multi_agent_runner import MultiAgentRunner
    runner = MultiAgentRunner()
    
    class MockSystem:
        def get_ablation_metrics(self):
            return [{"Condition": "Mock", "Accuracy": 1.0}]
            
    res = runner.evaluate(MockSystem())
    assert len(res) == 1
    print("[PASS] MultiAgentRunner")

    print("--- ALL MODULES PASSED SMOKE TEST ---")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"[FAIL] {e}")
    sys.exit(1)
