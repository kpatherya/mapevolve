# VLM failure benchmark and MapEvolve comparison

## 1. Claim and presentation flow

**Claim to test:** a VLM given an image and a natural-language question can often produce a plausible answer, but it is unreliable at identifying the place, selecting a comparable historical view, distinguishing visible evidence from outside facts, and refusing when the evidence is insufficient. MapEvolve is motivated as a question-specific, evidence-checking alternative—not as a claim that a VLM is never useful.

Present the story in this order:

1. **Motivation:** “What changed here?” sounds visual, but answering it requires identity, retrieval, viewpoint verification, and evidence.
2. **Closed-world challenge:** the model may use only the supplied image set and supplied records. This makes unsupported claims measurable.
3. **Naive VLM baseline:** send the image(s), question, and a fixed prompt directly to each model. Preserve the raw response.
4. **Failure gallery:** show confident wrong identities, false changes caused by viewpoint/lighting, invented historical facts, and answers to intentionally unanswerable questions.
5. **Four query classes:** simple, moderate, hard, hardest.
6. **MapEvolve teaser:** retrieve, verify, look up, calculate, cite evidence, and decline when a required step fails.
7. **Comparison:** quality, calibrated refusal, failure causes, latency, and cost—not only a single accuracy number.

## 2. Data inventory and split

The current corpus has 52 usable images in 9 building folders (55 files including 3 `.DS_Store` entries):

* **Pair-rich:** Tech Tower (17), Clough (11), Scheller Tower (8), Kendeda (6), McCamish Pavilion (4).
* **Pairable but small:** Brittain Dining Hall (2), Kessler Campanile (2).
* **Singletons:** Klaus (1), Scheller College (1).

Treat folder names as hidden ground truth, never as model input. Remove `.DS_Store`, assign stable image IDs, and record building, source filename, capture date when available, image dimensions, and camera coordinates when available. Do not infer a pair is comparable merely because it is in the same folder.

Use a **development set** for prompt and threshold design and a frozen **test set** for final results. Split by image, not by request, so that paraphrases or query variants never cross the split. Keep at least one held-out image for each pair-rich building. Report results separately for pair-rich, pairable-small, and singleton cases. Do not claim broad generalization from this corpus; call it a reviewed campus stress test.

Create reviewed annotations for:

* building identity and whether the image contains one or multiple buildings;
* image-pair relation: `comparable`, `same building/different view`, `different place`, or `uncertain`;
* visible attributes that can change (construction, facade, signage, trees, vehicles) versus nuisance differences (season, lighting, occlusion, crop);
* verified building facts and coordinates, with source and confidence;
* acceptable answer, required evidence, and correct refusal reason.

## 3. Query suite: four classes

Use 8–10 base cases per class, with 2–3 paraphrases each. This yields a compact, balanced suite while preventing one memorable prompt from deciding the result. Every case has a structured reference answer and an `answerable` flag.

### Class A — Simple: same building, same viewpoint

Purpose: test basic recognition and conservative visual description. Use reviewed near-duplicate or strongly aligned pairs where available.

Examples:

* “Are these photographs of the same Georgia Tech building? Name it and give two visible details that support your answer.”
* “What is unchanged between these two views? Mention only features visible in both.”
* “Which image shows the same facade as the reference: A, B, or neither? Explain using architectural landmarks.”
* “Is the apparent color difference a building change or likely illumination? Answer only from the images.”

Include hard negatives from visually similar campus buildings. Score identity, supporting observations, and whether the model invents an unseen detail.

### Class B — Moderate: same building, different viewpoint

Purpose: separate place recognition from viewpoint matching and expose false “change” claims.

Examples:

* “These may show the same building from different sides. What evidence supports or weakens that hypothesis?”
* “Align the views conceptually: which features should correspond, and which cannot be compared?”
* “Did the facade change, or did the camera move? Give a confidence level and list the strongest evidence.”
* “Find the historical image most suitable for comparing this reference. Reject candidates that show the building but not a compatible region.”

Require the model to distinguish `same building/different view` from `comparable`. A correct building with an invalid comparison is still a failure.

### Class C — Hard: investigative and record-dependent

Purpose: test whether the model knows that an image cannot establish an external fact.

Examples:

* “Who funded the creation of this building? If the supplied evidence does not establish it, say so and list the record needed.”
* “When was this building constructed or substantially renovated? Separate what is visible from what requires a source.”
* “What changed here over time, and which changes are supported by a comparable image versus a campus record?”
* “Does the image show the building’s original use? Give a sourced answer or decline.”
* “Find one documented reason for the building’s design, and distinguish it from your visual interpretation.”

Run each question in two conditions: **closed-world** (only the image set and provided records) and **bounded lookup** (a fixed allowlist of authoritative campus sources). This separates hallucination from retrieval failure. Require source URLs or record IDs in the answer; a plausible unsourced fact is incorrect.

### Class D — Hardest: multi-entity spatial reasoning

Purpose: test entity resolution, coordinate retrieval, tool use, and abstention under composition.

Examples:

* “Which of these two buildings is closer to Klaus? Identify both buildings first, then compare their coordinates.”
* “Which candidate is closer to the Student Center? Do not estimate from pixel size; use verified building locations.”
* “Place these three buildings in nearest-to-farthest order from Tech Tower, and state what evidence supports the ordering.”
* “Which building is farther from Klaus but appears larger in the photo? Explain why apparent size is not distance.”
* “If one image contains two buildings, identify the pair and say which one is closer to the named landmark. Decline if either identity is uncertain.”

Use coordinate-based straight-line distance as the primary reproducible target; optionally report walking distance as a separate task only when a fixed routing source is available. The model must not receive folder names or coordinates in the prompt.

## 4. Baselines and experimental matrix

Run every case with the same image ordering, resolution policy, system prompt, and output schema across:

1. **Direct VLM baseline:** image(s) + query; no retrieval, tools, or hidden labels.
2. **Direct VLM with explicit abstention instruction:** identical inputs, but require `answer`, `evidence`, `confidence`, and `cannot_answer_reason`.
3. **Oracle-evidence ablation:** provide the reviewed relevant pair/record to measure answer-generation quality separately from retrieval.
4. **MapEvolve prototype:** question router → image retrieval → identity/pair verification → record lookup or coordinate calculation → evidence-constrained answer.

Evaluate the four named model slots from the presentation plan—DeepSeek V4.1 Flash, Gemini 3.8 Flash, Claude Opus 5.5, and GPT-6 Luna—through OpenRouter. Before running, resolve each human-readable name to the exact OpenRouter model ID, record the ID, provider, date, pricing, context limit, and any vision/tool limitations. If a name is unavailable, mark it unavailable; do not silently substitute a different model.

Pin temperature, max output tokens, reasoning settings, image detail/size, timeout, retry policy, and seed where supported. Run at least three independent repetitions per case for stochastic models. Use one request per case, then a separate judge pass; do not let a judge model rewrite the original answer.

## 5. OpenRouter runner and reproducibility

Implement a small runner with:

* manifest-driven cases (`case_id`, class, image IDs, prompt, allowed evidence, reference answer, answerability);
* an adapter that converts local images to the OpenRouter-supported image payload;
* environment-based credentials loaded from `.keys` locally but never committed, printed, or copied into logs;
* raw request metadata and raw response saved with secrets and authorization headers redacted;
* exponential backoff for transient errors, bounded retries, and an explicit `api_error` outcome;
* request hash, model ID, timestamp, latency, token usage, estimated cost, status, and cache-hit status;
* schema validation for model output, with malformed output counted as a failure rather than repaired silently.

Cache by `(model_id, prompt_hash, image_hashes, generation_settings)` only for a declared deterministic/cache condition. Run a no-cache condition for latency and cost comparisons. Never compare cached latency with uncached latency.

## 6. Scoring and failure analysis

Use a structured rubric rather than relying on judge-model preference:

* **Identity accuracy:** exact building, partial/ambiguous, wrong.
* **Pair decision:** comparable, same-building/different-view, different, uncertain.
* **Claim support:** every material claim is supported by an image or named record.
* **Answer correctness:** 0–4 points using case-specific reference criteria.
* **Refusal calibration:** correct answer, justified refusal, unnecessary refusal, unsupported answer.
* **Confidence calibration:** confidence should fall on ambiguous, singleton, and unanswerable cases.
* **Evidence quality:** correct evidence selected and limitations stated.

Report per class and overall: accuracy, macro-F1 where applicable, answer coverage, unsupported-claim rate, correct-refusal rate, over-refusal rate, calibration/error curves, median and p95 latency, cost per request, and API-error rate. Include bootstrap confidence intervals over cases; do not treat paraphrases as independent samples in the primary interval.

Assign each failure a primary cause:

1. recognition/entity resolution;
2. retrieval/ranking;
3. viewpoint or geometric verification;
4. record/source lookup;
5. spatial calculation/tool use;
6. evidence attribution;
7. hallucination or overconfident refusal;
8. generation/format/API failure.

Have a second human rater score at least 20% of cases, stratified by class and refusal status. Report agreement before adjudication. Preserve representative raw failures and a short “why the model was tempted” note for the presentation.

## 7. Results that motivate MapEvolve

The central figure should be a **failure matrix**: rows are failure causes, columns are models and MapEvolve, with counts and rates by query class. Pair it with:

* an accuracy-versus-coverage plot, so “refuse everything” cannot win;
* a cost/latency-versus-supported-answer plot;
* a four-panel qualitative gallery showing one failure and one recovery per class;
* an ablation showing whether MapEvolve gains come from retrieval, verification, records, routing, or refusal—not merely from using a different generator.

The conclusion should be bounded: on this campus corpus, direct VLMs are useful recognizers and answer generators but fail systematically when the task requires evidence selection, comparable-view verification, external facts, or multi-step spatial grounding. MapEvolve’s contribution is the workflow and measurement discipline that makes those failure modes visible and actionable.

## 8. Immediate implementation order

1. Build the image manifest and reviewed pair/record annotation sheet.
2. Generate the four-class case manifest, including explicit unanswerable and hard-negative cases.
3. Implement the direct OpenRouter runner and schema validation; run a 2-case smoke test per model.
4. Freeze prompts and model IDs, then run development cases and revise only the rubric/router.
5. Freeze the test set and execute all model repetitions with raw logs.
6. Add retrieval, verification, records, and coordinate calculation one component at a time.
7. Score, audit disagreements, produce the failure matrix, and select the presentation gallery.
