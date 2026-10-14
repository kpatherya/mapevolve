# Qualitative failures

Selection rule, applied in order to scored responses that are not labeled correct:

1. Hallucinated structural-change claims on medium or hard cases, lowest final score first.
2. Remaining hard-case responses, lowest final score first.
3. Remaining medium-case responses, lowest final score first.
4. Prefer a model and case that have not already been selected.

The examples do not represent all model behavior.

## Example 1

- Case: H1 (hard)
- Model: KIMI (`moonshotai/kimi-k2.5`)

Ground truth:

```text
H1 — Tech Tower, Hard, Long-Term Change
Question: “What changed between these two images?”
Ground truth: The same central building remains, but the surrounding campus environment has changed substantially. The 2018 image has much more mature vegetation and different landscaping, paths, and surrounding structures. Some structures visible in the 1918 scene are no longer present, while newer campus elements appear in the modern image. Differences in color should not be treated as architectural change because the 1918 photograph is monochrome/sepia.
```

Model response:

```text
The tower architecture changed significantly from an ornate crenellated design with a clock in Image B to a simpler pointed spire in Image A. A tall brick chimney visible to the right of the main building in Image A is entirely absent in Image B. The main building's facade, window arrangements, and foreground landscape grading also differ substantially between the two images.
```

- Judge scores: core=0, confounder=0, hallucination_free=0, hallucinated_change=True, raw_total=0, final_total=0, label=incorrect
- Unsupported points: The tower architecture changed significantly from an ornate crenellated design with a clock to a simpler pointed spire.; The main building's facade and window arrangements differ substantially between the two images.; A tall brick chimney visible in one image is entirely absent in the other (treated as a key change rather than part of broader campus evolution).
- Missing points: The central historic building remains recognizable across both images.; The surrounding campus environment changes substantially.; Vegetation is much more mature in the modern image.; Landscaping and path layout differ.; Some surrounding structures disappear and/or newer elements appear.
- Interpretation: The response asserts a physical or structural change that the answer key does not support: The tower architecture changed significantly from an ornate crenellated design with a clock to a simpler pointed spire.

## Example 2

- Case: H2 (hard)
- Model: MISTRAL (`mistralai/mistral-small-3.2-24b-instruct`)

Ground truth:

```text
H2 — Carnegie Building, Hard, Subtle Restoration
Question: “Compare these two images of the same building. What changed?”
Ground truth: The overall structure and architectural form remain the same. The later image reflects restoration work, including changes to windows/doors and restoration of historic exterior details. The masonry/exterior also appears cleaner. The model should not claim that major structural elements such as the columns, roofline, or building footprint changed.
```

Model response:

```text
The most notable differences between the two images are the environmental and seasonal conditions. In Image A, the weather is clear and sunny, with bright lighting and green vegetation, while in Image B, the sky is overcast, and the trees are bare, indicating a different season, likely winter. Additionally, the flowers in the foreground of Image B are not visible in Image A, suggesting a seasonal change. There are no visible structural or physical changes to the building itself.
```

- Judge scores: core=0, confounder=1, hallucination_free=0, hallucinated_change=True, raw_total=1, final_total=1, label=incorrect
- Unsupported points: Specific claims about clear and sunny vs. overcast weather.; Claims about green vegetation vs. bare trees and missing flowers indicating seasonal change.
- Missing points: Restoration work to windows/doors and historic exterior details.; Exterior/masonry appears cleaner or restored.; Overall structure remains the same but with subtle restoration changes.
- Interpretation: The response asserts a physical or structural change that the answer key does not support: Specific claims about clear and sunny vs. overcast weather.

## Example 3

- Case: M2 (medium)
- Model: GEMINI (`google/gemini-3.1-pro-preview`)

Ground truth:

```text
M2 — Kessler Campanile Fountain, Medium, Operational + Transient Change
Question: “What changed between these two images?”
Ground truth: The fountain is filled and operating with water jets in the first image, while it is empty or turned off in the second. The second image also contains a large crowd. No clear structural change to the Campanile or fountain structure is visible.
```

Model response:

```text
The most prominent physical difference is the state of the fountain; in Image A, it is filled with water with active spraying jets, whereas in Image B, the basin is completely dry. Additionally, the monument in Image B features a distinct white band around its lower section that is not present in Image A. A major temporary difference is the large crowd of people gathered in the background and foreground of Image B, while Image A is devoid of people.
```

- Judge scores: core=1, confounder=0, hallucination_free=0, hallucinated_change=True, raw_total=1, final_total=1, label=incorrect
- Unsupported points: The monument in Image B features a distinct white band around its lower section that is not present in Image A
- Missing points: No clear structural change to the fountain/Campanile is visible
- Interpretation: The response asserts a physical or structural change that the answer key does not support: The monument in Image B features a distinct white band around its lower section that is not present in Image A.

## Example 4

- Case: M1 (medium)
- Model: CLAUDE (`anthropic/claude-sonnet-4`)

Ground truth:

```text
M1 — Klaus Stairs, Medium, Real Change + Environmental Change
Question: “What changed between these two images?”
Ground truth: The staircase has been painted in multiple colors in the second image. The second image also shows snow and winter conditions. The building entrance and main structural elements appear unchanged.
Expected behavior: Identify the painted staircase as a real visible change, mention snow/winter as a temporary environmental difference, and avoid claiming unsupported structural modifications.
```

Model response:

```text
The most significant change between these two images is that the stairs have been painted with rainbow/pride flag colors in Image B, while they appear to be plain concrete/stone colored in Image A. Additionally, Image B shows snow coverage on the ground and stairs, while Image A shows clear, dry conditions with green foliage. The season has also changed from what appears to be summer/fall (with lush green trees) in Image A to winter (with bare tree branches) in Image B.

The rainbow painting of the stairs represents a permanent structural/decorative change to the building's entrance, while the snow and seasonal vegetation differences are temporary environmental changes.
```

- Judge scores: core=2, confounder=1, hallucination_free=0, hallucinated_change=True, raw_total=3, final_total=3, label=partially_correct
- Unsupported points: The rainbow painting of the stairs represents a permanent structural/decorative change to the building's entrance.
- Missing points: Main building entrance/structure appears unchanged.
- Interpretation: The response asserts a physical or structural change that the answer key does not support: The rainbow painting of the stairs represents a permanent structural/decorative change to the building's entrance.
