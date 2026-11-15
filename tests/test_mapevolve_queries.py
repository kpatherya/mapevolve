"""Check benchmark integrity without calling model APIs."""
import json
from collections import Counter
from PIL import Image
from jev_router.dataset import ROOT, load_manifest, routing_inputs, validate


def test_mapevolve_coverage_and_review_status():
    cases = load_manifest(ROOT / 'datasets/mapevolve_images.json')
    assert validate(cases) == []
    assert Counter(c.query_type for c in cases) == {
        'landmark_identification': 50, 'factual_retrieval': 50}
    for typ in ('landmark_identification', 'factual_retrieval'):
        assert {c.difficulty for c in cases if c.query_type == typ} == {'easy','moderate','hard'}
    assert all(c.split == 'test' and c.labels['reviewed'] is False for c in cases)
    assert Counter(c.tags[0] for c in cases if c.query_type == 'factual_retrieval') == {
        'building_record':30, 'capture_metadata':12, 'image_only_unanswerable':8}


def test_no_hidden_metadata_or_answer_fields_in_inputs():
    cases = load_manifest(ROOT / 'datasets/mapevolve_images.json')
    for c in cases:
        inp = routing_inputs(c)
        assert set(inp) == {'question','images','supplied_evidence'}
        for p in inp['images']:
            assert p.name.startswith('image_')
            with Image.open(p) as im:
                assert not im.getexif()
                assert max(im.size) <= 2048
        if c.query_type == 'landmark_identification' or 'image_only_unanswerable' in c.tags:
            assert inp['supplied_evidence'] is None


def test_capture_answers_use_original_exif_not_filename():
    inventory = json.loads((ROOT/'data/mapevolve_queries/image_inventory.json').read_text())
    originals = {r['path']:r for r in inventory}
    cases = load_manifest(ROOT / 'datasets/mapevolve_images.json')
    mismatches = 0
    for c in cases:
        if 'capture_metadata' not in c.tags:
            continue
        p = ROOT / c.labels['source_image']
        with Image.open(p) as im:
            actual = im.getexif().get_ifd(34665)[36867]
        assert actual == originals[c.labels['source_image']]['metadata']['DateTimeOriginal']
        assert actual in c.supplied_evidence
        assert actual[:4] in c.reference['answer']
        if p.name[:2].isdigit() and int(p.name[:2]) != int(actual[:4]) % 100:
            mismatches += 1
    assert mismatches >= 4
