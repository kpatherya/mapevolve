# Smoke test

E1 uses the fixed benchmark prompt. The earlier two-image check only asked for the word OK and is not scored.

## Availability

| Slot | Requested | Resolved | Backend | Image | Two-image smoke | Status | Substitution |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OPENAI_FRONTIER_VISION | `gpt-5.1` | `gpt-5.1-2025-11-13` | openai | True | pass | ready | API returned snapshot id gpt-5.1-2025-11-13 for requested gpt-5.1. |
| GEMINI | `google/gemini-3.1-pro-preview` | `google/gemini-3.1-pro-preview` | openrouter | True | pass | ready |  |
| CLAUDE | `anthropic/claude-sonnet-4` | `anthropic/claude-sonnet-4` | openrouter | True | pass | ready |  |
| QWEN | `qwen/qwen3-vl-235b-a22b-instruct` | `qwen/qwen3-vl-235b-a22b-instruct` | openrouter | True | pass | ready |  |
| GEMMA | `google/gemma-3-27b-it` | `google/gemma-3-27b-it` | openrouter | True | pass | ready |  |
| LLAMA | `meta-llama/llama-4-maverick` | `meta-llama/llama-4-maverick` | openrouter | True | pass | ready |  |
| MISTRAL | `mistralai/mistral-small-3.2-24b-instruct` | `mistralai/mistral-small-3.2-24b-instruct` | openrouter | True | pass | ready |  |
| KIMI | `moonshotai/kimi-vl-a3b-thinking` | `moonshotai/kimi-k2.5` | openrouter | True | pass | substituted | exact id moonshotai/kimi-vl-a3b-thinking is unavailable or not image-capable; using closest current image-capable KIMI family model moonshotai/kimi-k2.5 |

## E1 responses

### OPENAI_FRONTIER_VISION

- Status: success
- Resolved model: `gpt-5.1-2025-11-13`
- Latency seconds: 3.4840333139291033
- Reported cost USD: None
- Parameter adjustments: ['retry 1 after status None: Request timed out.']

There is no visible structural change to the building, paths, lights, or general layout; they appear the same in both images. The main differences are due to framing and environment: Image A is taken closer to the flower bed and lower to the ground, while Image B is taken farther back and higher, showing more of the surrounding walkway and park. Image B also includes a person walking on the path, which is not present in Image A.

### GEMINI

- Status: success
- Resolved model: `google/gemini-3.1-pro-preview`
- Latency seconds: 21.522544921957888
- Reported cost USD: 0.026320000000000003
- Parameter adjustments: ['increased max_tokens from 350 to 4096 after the output was empty or truncated']

Based on the images provided, there are no visible structural or physical changes to the location itself. The primary difference is the framing and perspective; Image B was taken with a much wider-angle lens, which reveals more of the paved pathways, the surrounding trees, and an adjacent building on the far left. The only other notable difference is the temporary presence of a person walking in the midground of Image B.

### CLAUDE

- Status: success
- Resolved model: `anthropic/claude-sonnet-4`
- Latency seconds: 8.011221266933717
- Reported cost USD: 0.012414
- Parameter adjustments: none

Looking at these two images of the same campus location, I can identify several differences:

The most notable change is in the flower beds - Image A shows the yellow flowers arranged in a more linear, rectangular pattern, while Image B shows them in curved, flowing beds that follow the natural contours of the landscape. The pathways and walkway configuration also appear different, with Image B showing more curved pathways compared to the more geometric layout in Image A.

Additionally, there's a person visible walking in the middle area of Image B who is not present in Image A, and the camera angle/framing is slightly different between the two shots, with Image B taken from a somewhat different vantage point that shows more of the curved flower bed design.

The building, trees, and overall landscape structure remain the same between both images.

### QWEN

- Status: success
- Resolved model: `qwen/qwen3-vl-235b-a22b-instruct`
- Latency seconds: 37.27507923298981
- Reported cost USD: 0.00132884
- Parameter adjustments: none

The most significant visible difference is the presence of a person walking on the path in Image B, who is absent in Image A. The framing is also slightly wider in Image B, revealing more of the surrounding trees and pathways on the left and right. The lighting and shadows appear nearly identical, suggesting the photos were taken very close in time. No persistent physical or structural changes to the building or landscape are visible.

### GEMMA

- Status: success
- Resolved model: `google/gemma-3-27b-it`
- Latency seconds: 5.385004887008108
- Reported cost USD: 8.771e-05
- Parameter adjustments: none

The most noticeable difference between the two images is the presence of a person walking on the path in Image B, who is absent in Image A. Additionally, the lighting appears slightly different, with Image B appearing a bit brighter overall. The framing is also subtly altered, with Image B capturing a wider view of the surrounding landscape. No persistent physical or structural changes are visible between the two images.

### LLAMA

- Status: success
- Resolved model: `meta-llama/llama-4-maverick`
- Latency seconds: 12.746160599053837
- Reported cost USD: 0.0007922175
- Parameter adjustments: none

The most notable difference between the two images is the camera's perspective and framing. Image B is taken from a slightly different angle, capturing a wider view of the surrounding area, including more of the walkway and the trees on the left side. The composition of the images differs, with Image A focusing more on the building and Image B showing more of the foreground flowers and the path. No other significant structural or environmental changes are visible between the two images.

### MISTRAL

- Status: success
- Resolved model: `mistralai/mistral-small-3.2-24b-instruct`
- Latency seconds: 4.750176909030415
- Reported cost USD: 0.000384525
- Parameter adjustments: none

The most notable difference between the two images is the presence of a person in Image B, who is walking in the garden area, while Image A does not show any people. Additionally, the lighting and shadows suggest that the images were taken at slightly different times of day, with Image B having slightly longer shadows. The vegetation, particularly the yellow flowers, appears to be in the same state in both images, indicating no significant change in the plant life. There are no visible structural or physical changes to the building or the surrounding area between the two images.

### KIMI

- Status: success
- Resolved model: `moonshotai/kimi-k2.5`
- Latency seconds: 175.07877879205626
- Reported cost USD: 0.017223899
- Parameter adjustments: ['increased max_tokens from 350 to 4096 after the output was empty or truncated']

Image B shows a new circular paved plaza with brick patterns in the middle ground and a bench on the right that are absent in Image A. The trees display seasonal progression with fuller spring foliage, and a person is present in Image B. Image B is captured from a wider angle positioned further left.
