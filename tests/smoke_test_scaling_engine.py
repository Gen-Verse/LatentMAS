import sys
import torch
import logging

logging.basicConfig(level=logging.INFO)

print("--- SMOKE TEST: Multilingual Scaling Engine ---")

# NOTE: this smoke test used to target a phantom API (hub.add_language_adapter,
# GeometryConditionedCVAEPrior) copied from the strategy documents rather than the
# ported code — it failed at step 1 on every run. It now exercises the real classes.
try:
    print("1. Testing UniversalLatentHub...")
    from latent_coordination.latent_space.universal_space import UniversalLatentHub
    hub = UniversalLatentHub(universal_dim=64)
    hub.register_agent("agent_th", hidden_dim=128)
    hub.register_agent("agent_en", hidden_dim=96)
    assert hub.is_registered("agent_th") and hub.is_registered("agent_en")
    states = torch.randn(2, 128)
    received = hub.transfer("agent_th", "agent_en", states)
    assert received.shape == (2, 96)
    print("[PASS] UniversalLatentHub")

    print("2. Testing RecursiveLatentCore...")
    from latent_coordination.latent_space.recursive_core import RecursiveLatentCore
    core = RecursiveLatentCore()
    z_init = torch.randn(1, 512)
    z_out = core(z_init)
    assert z_out.shape == (1, 512)
    print("[PASS] RecursiveLatentCore")

    print("3. Testing CVAETopologyPrior...")
    from latent_coordination.topology.cvae_prior import CVAETopologyPrior, TrainingConfig
    cfg = TrainingConfig(z_dim=16, query_dim=64, max_n_agents=3)
    prior = CVAETopologyPrior(cfg)
    G = torch.zeros(2, 3, 3)
    Q = torch.randint(0, cfg.query_vocab_size, (2, 16))
    recon_G, mu, logvar = prior(G, Q)
    assert recon_G.shape == (2, 3, 3)
    assert mu.shape == (2, 16) and logvar.shape == (2, 16)
    print("[PASS] CVAETopologyPrior")

    print("4. Testing QueryReconstructionProbe...")
    from latent_coordination.eval.verification_probe import QueryReconstructionProbe, LatentDriftException
    probe = QueryReconstructionProbe()
    # The probe refuses to gate on an untrained decoder — fit on a small real pair
    # batch first (here: random smoke-test tensors standing in for hub states).
    fit_states = torch.randn(16, 512)
    fit_queries = torch.randn(16, 1024)
    probe.fit_decoder(fit_states, fit_queries, n_epochs=5)
    z_t = torch.randn(1, 512)
    q_orig = torch.randn(1, 1024)
    try:
        drift_score = probe(z_t, q_orig)
    except LatentDriftException:
        pass # Expected or possible depending on random init
    print("[PASS] QueryReconstructionProbe")

    print("5. Testing MultiAgentRunner (Dry Run)...")
    import tempfile, os
    from latent_coordination.eval.multi_agent_runner import MultiAgentRunner
    runner = MultiAgentRunner()

    class DryRunSystem:
        def get_ablation_metrics(self):
            return [{"Condition": "DryRun", "Accuracy": 1.0}]

    out_path = os.path.join(tempfile.mkdtemp(), "final_report.json")
    res = runner.evaluate(DryRunSystem(), output_path=out_path)
    assert len(res) == 1 and os.path.exists(out_path)
    print("[PASS] MultiAgentRunner")

    print("--- ALL MODULES PASSED SMOKE TEST ---")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"[FAIL] {e}")
    sys.exit(1)
