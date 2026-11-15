# MapEvolve image benchmark

100 cases drawn from the supplied `MapEvolve_Images` folder, with answer keys,
source records, and explicit grading criteria. Labels were drafted by Codex
after inspecting the images; **human review is pending**. No model calls or
paid benchmarks were run to create this set.

## Contents

- [Queries with inline images and answers](queries_and_answers.md)
- [Spreadsheet answer key](answer_key.csv): includes questions, paths, difficulty,
  supplied evidence, expected answers, required points and prohibited claims.
- [Runnable schema-v2 manifest](../../datasets/mapevolve_images.json)
- [Source cards and URLs](sources.json): paraphrases of official Georgia Tech records.
- [Image inventory](image_inventory.json): original paths, dimensions, selected
  metadata, exported paths and original SHA-256 hashes for selected images.
- `images/`: 51 selected JPEG copies, rotated using EXIF orientation, limited to
  a 2048-pixel longest edge, with neutral filenames and no EXIF metadata.

Original files are unchanged. HEIC files and unidentified/unneeded photos were
excluded. Each case uses one image; 100 cases reuse 51 photographs across
identification and factual tasks. These are not 100 independent landmarks.

## Coverage

| Prompt type | Easy | Moderate | Hard | Total |
| --- | ---: | ---: | ---: | ---: |
| Landmark identification | 15 | 20 | 15 | 50 |
| Factual retrieval | 17 | 15 | 18 | 50 |

Landmarks include Tech Tower, Clough Commons, Kendeda, Kessler Campanile,
Klaus, McCamish Pavilion, Brittain Dining Hall, Carnegie, the older Scheller
College building, Scheller Tower and George Tower. One close-up showing only
a GT logo deliberately lacks enough evidence for an exact identification.
In particular, the source folder named `Scheller_Tower` contains views of
George Tower too: selected labels follow the visible subject, not the folder.

Difficulty is a provisional editorial label. For identification it reflects
readable signage, viewpoint, cropping, lighting, foliage and interior ambiguity.
For factual tasks it reflects single-fact lookup, multi-fact synthesis,
conflicting date concepts and recognizing insufficient evidence. It is not a
measured model-performance tier. Highly ambiguous identities and all historical
facts should be checked by a campus expert before final scoring.

## What the factual tasks measure

| Track | Cases | Evidence given to the model | Expected behavior |
| --- | ---: | --- | --- |
| `building_record` | 30 | A cited, paraphrased official building record | Retrieve the requested facts or give a short supported description |
| `capture_metadata` | 12 | Original image's EXIF DateTimeOriginal, sometimes its original filename | Report the recorded date; distinguish it from filenames or opening years |
| `image_only_unanswerable` | 8 | None | Explain the missing evidence and request metadata or relevant records |

This follows the repository's **closed-world** answerability convention:
facts must be supported by the image or supplied evidence. It is not a test of
web-search capability or memorized building history. The source cards emulate
retrieved documents; they do not exercise a retrieval engine. Report results
for these three factual tracks separately, rather than interpreting abstention
success as factual lookup accuracy. All landmark cases receive images only.

Camera EXIF is a recorded claim, not independently authenticated truth. Several
filename prefixes disagree with capture years: e.g. Kendeda `20C...` records
2019; Scheller Tower `26-...` records 2025; McCamish `13C...` records 2012.
The answer keys use DateTimeOriginal, never a year guessed from the filename.
The image-only controls have neutral filenames and stripped metadata so they
cannot silently leak those dates to classifiers or VLMs.

## Scoring

Use the manifest's `reference.required_points` and
`reference.unsupported_claims_to_avoid` with the existing 0–4 judge rubric:
4 = covers the required points without unsupported claims; 3 = minor omission;
2 = partial; 1 = mostly wrong; 0 = wrong or confidently invents an unavailable
fact. Equivalent wording and the listed identification aliases are acceptable.
A visual explanation is optional unless a required point asks for it. Hard,
obscured identifications may be phrased tentatively if the correct name is given.

For unanswerable controls, full credit means acknowledging the missing evidence
and identifying a suitable record/tool to resolve it; a guessed date or cost is
not correct merely because it matches hidden metadata. Never send the answer
key, labels, inventory or reference fields to the model as prompt context.
`routing_inputs` already restricts model inputs to images, question and supplied
evidence. Suggested reports: score by prompt type and difficulty, by factual
track, unsupported-claim rate, mean latency, and total/mean cost. Use the same
cases and model settings for every strategy.

All cases are marked `split=test`. Keep them out of router tuning and quality
predictor training when measuring held-out performance. The new set shares
buildings and some original/near-duplicate images with the older Georgia Tech
pilot; holding this file out does not establish independence from that pilot.
Do not report all 100 cases as independent samples; report location-level
averages as well as per-query averages.

## Run after reviewing the answer key

The [pre-benchmark audit and frozen configuration](../../benchmark/frozen/mapevolve_v1/AUDIT.md)
define the proposed three-strategy comparison. Verify them with
`python -m scripts.verify_mapevolve_freeze`. The commands below demonstrate the
existing all-model collector; they do not automatically load the frozen settings.

From the repository root, with the project dependencies installed:

```bash
# Plan only; this command makes no paid model calls.
python -m scripts.run_live_benchmark --dataset mapevolve_images --dry-run

# Example collection: change model keys and the cost cap as desired.
python -m scripts.run_live_benchmark --dataset mapevolve_images \
  --models GEMMA OPENAI_FRONTIER_VISION --run-id mapevolve-images-v1 --max-cost 1.00

# Grade collected answers using the existing rubric (a paid judge call).
python -m scripts.grade_responses judge \
  results/live_runs/mapevolve-images-v1.jsonl --backend openai --model gpt-5.1
```

The runner collects answers for the chosen models. The existing offline
evaluator refits routing scores from collected grades, so it must not be used
unchanged to measure the frozen pre-existing router. Preserve a copy of the
frozen manifest and registry for each benchmark round. Estimates and the
runner's cost cap are approximate, not a provider billing guarantee.

To rebuild the exports from the inventory and manually authored case definitions:

```bash
python -m scripts.build_mapevolve_queries
```

Review edits belong in `scripts/build_mapevolve_queries.py`; rerunning it
regenerates the manifest, CSV, Markdown and neutral images. Set human review
flags in the final manifest only after review (a rebuild resets draft flags).
