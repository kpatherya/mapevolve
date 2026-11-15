"""Verify and load the frozen comparison inputs without making API calls.

    python -m scripts.verify_mapevolve_freeze

A batch comparison runner can call load_frozen() to use these settings instead
of mutable app defaults. The stock Streamlit app does not load this snapshot.
"""
from pathlib import Path
import hashlib
import json
import yaml
from jev_router.dataset import ROOT, load_manifest, validate
from jev_router.registry import ModelRegistry
from jev_router.policy import RoutingPolicy

FREEZE = ROOT / 'benchmark/frozen/mapevolve_v1'


def load_frozen():
    lock = json.loads((FREEZE/'lock.json').read_text())
    changed = [name for name, digest in lock['sha256'].items()
               if not (ROOT/name).is_file()
               or hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest]
    if changed:
        raise ValueError('Frozen benchmark inputs changed: '+', '.join(changed))
    cfg = yaml.safe_load((FREEZE/'routing.yaml').read_text())
    protocol = json.loads((FREEZE/'protocol.json').read_text())
    cases = load_manifest(FREEZE/'cases.json')
    errors = validate(cases)
    if errors:
        raise ValueError(str(errors))
    assert len(cases) == 100
    assert cfg['generation']['temperature'] == 0
    assert cfg['generation']['max_failover_models'] == 0
    assert cfg['jev']['image_aware'] is True
    assert cfg['cascade']['enabled'] is False
    assert cfg['pareto']['prune_dominated'] is False
    assert cfg['quality_model']['path'] is None
    assert protocol['adaptive_mode'] == 'balanced'
    registry = ModelRegistry.load(FREEZE/'model_registry.json', routing=cfg)
    policy = RoutingPolicy(registry, cfg, use_quality_model=False)
    assert policy.quality_model is None
    for key in protocol['allowed_model_keys']:
        assert registry.is_listed(key) and registry.supports_images(key, 1)
        assert registry.general_quality(key) is not None
    assert policy.cheapest_eligible(1, allowed_keys=protocol['allowed_model_keys']) == protocol['cheapest_baseline']
    assert policy.strongest(1, allowed_keys=protocol['allowed_model_keys']) == protocol['strongest_baseline']
    return cfg, registry, cases, protocol


def main():
    cfg, registry, cases, protocol = load_frozen()
    print(f'Freeze verified: {len(cases)} cases; {len(protocol["allowed_model_keys"])} allowed models.')
    print('Baselines:',protocol['cheapest_baseline'], '/', protocol['strongest_baseline'])
    print('Adaptive: balanced; Jev image-aware; cascade/learned predictor/pruning/failover disabled.')
    print('Answer key: source-checked by AI; human review pending. See AUDIT.md for limitations.')


if __name__ == '__main__': main()
