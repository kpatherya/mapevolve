# T02 — Query classifier

**Depends on:** T01  
**Goal:** Classify text, validate evidence, and produce non-execution actions.

## Scope

- Add a deterministic, rule-based classifier for:
  `same_building_verification`, `change_captioning`,
  `evidence_backed_investigation`, `proximity_computation`, and `unknown`.
- Keep classification independent of image count and folder names.
- Require exactly two images for `same_building_verification`.
- Return `request_evidence` for missing or incorrect evidence.
- Return `abstain` with
  `reason_code="capability_not_yet_supported"` for each recognized family
  other than `same_building_verification`.
- Return a structured abstention for `unknown`; never execute it.
- Record a rule identifier and test representative positive, negative, and
  ambiguous/unknown phrases with `unittest`.

## Acceptance

- All four families remain in the schema and taxonomy.
- Non-execution actions never invoke a backend.
- The same query gets the same family with zero, one, or two images.
