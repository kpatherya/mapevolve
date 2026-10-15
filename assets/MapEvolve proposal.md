**MapEvolve: Reliable Answers About How Places Change**

CS 6220: Big Data Systems and Analytics  
Group 1: Kausar Patherya, Alejandro Hernandez, Şamil Namli, Shaktik Bhattacharyya

MapEvolve helps people understand the campus around them by connecting a photo they take today with older photographs, building information, and campus information.

**\~**

**Motivation and objectives**

A photograph can show what a campus building looks like today, but it does not by itself identify the building name, how it looked in the past, and where it sits relative to other campus landmarks. Answering these questions requires connecting the photograph to building records, relevant older images, and location information. MapEvolve will bring these sources together so people can explore Georgia Tech’s campus starting with a photo and a natural-language query.

To illustrate, a user will take a picture of the Klaus building and ask: “What changed here?” MapEvolve will find an older photograph of the same place, check whether the camera angles support a comparison, and describe the visible changes. The retrieved images would be geo-tagged, revealing the building’s location. Additionally, if provided a picture of two buildings, and prompted with “Which of these two buildings is closer to the Student Center?”, MapEvolve would (1) identify the two buildings and extract their coordinates, (2) run a web lookup on the Student Center and find its location, and (3) return the closest building to the landmark.

The central challenge is finding evidence that actually supports the question. The key bottleneck is information access, not reasoning. Two buildings may look similar, and photographs of the same place may show different sides or hide important details. A difference in lighting, trees, or camera position should not be mistaken for a change to the building. Likewise, a photograph alone does not contain facts such as the building’s construction date or the reason for a renovation. **MapEvolve will distinguish what the images show from what recorded information establishes.** When it cannot identify the place or verify a comparison, it will explain the limitation rather than invent an answer.

We have begun with a small, checked collection of older campus photographs, new photographs taken from similar locations, along with building and location records. This catalog will be expanded as the system develops. We will evaluate whether MapEvolve identifies the correct buildings, finds usable historical comparisons, and gives answers supported by the available evidence. We will also measure how often it correctly declines to answer, how quickly it responds, and the cost of a request.

**Related work**

Understanding how a campus has changed requires more than recognizing the details of a single landmark image provided by the user. It requires finding earlier views, deciding whether they support a comparison, and connecting visible details to reliable records. Several research areas provide foundations for this workflow, but each addresses a different part of the problem.

**Historical photography and place recognition** provide a starting point. Computational rephotography helps photographers recreate the viewpoint of an earlier image.1 This matters because a different camera position can make the same landmark look different. Place-recognition research tackles the challenge of finding a matching landmark within an image collection: NetVLAD learns whole-image representations for large-scale place search.2 Benchmarks such as CMU Seasons test localization under changing foliage, weather, and illumination.3 Together, these areas motivate collecting repeat photographs while making older images searchable despite changes in appearance.

**Local matching and change captioning** address what happens after retrieval. SuperGlue matches local image features and rejects points without suitable correspondences.4 Such matches can support a same-scene check, but identifying an identical landmark is not the same as describing changes in its facade. Change-captioning research then asks how to describe differences between images. Robust change captioning distinguishes meaningful scene changes from distractors such as viewpoint shifts.5 MapEvolve must address an additional problem: it receives a new photograph and a question, rather than an already selected comparison pair. The image-matching process must support change reporting.

**Retrieval-augmented answering and visual agents** connect these components to user questions. Retrieval-augmented generation (RAG) combines language generation with retrieved external passages.6 More recent work, including Visual-RAG, investigates answering questions using retrieved images as visual evidence.7 Visual programming shows how natural-language instructions can become modular programs that involve visual models and other functions.8 Different steps must be followed for different questions rather than running every component for every request.

MapEvolve brings these ideas together in a small, reviewed campus photograph collection. A change question requires historical image retrieval and viewpoint verification; a building question requires identification and recorded facts; a proximity question requires verified identities, coordinates, and distance calculation. Our main contribution is an evaluated integration that distinguishes visual observations from documented facts, declines unsupported comparisons, and measures whether question-specific workflows reduce latency and cost without sacrificing answer quality.

**Proposed work**

MapEvolve will connect campus photographs, building records, and location information through two stages: offline preparation and online question answering. The goal is to find evidence that supports the user’s question, verify it, and answer only what the evidence establishes.

**![][image1]**

**Offline preparation.** We will begin with a small, reviewed collection of historical and present-day campus photographs. Each image will receive a stable ID and record containing the building name, capture date, source and camera location when available. Building coordinates will be stored separately from camera coordinates. The system will compute CLIP image embeddings and store them in a searchable index. Building names, descriptions, facts and coordinates will support text search. Reusable features from POLARIS may be incorporated in the image-matching stage.9 Offline preparation will re-run when records are added or updated, rather than for every request.

**Question-specific execution.** At query time, MapEvolve will receive an image and a question. One of three workflows could be followed:

* *What changed here?* Retrieve historical candidates, verify building identity and view compatibility, and describe only differences supported by comparable images.  
* *Tell me about this building.* Identify the building and retrieve supporting photographs and records, separating visible observations from documented facts.  
* *Which building is closer to the Student Center?* Identify both buildings, retrieve verified coordinates, and calculate walking distance to the landmark.

Image similarity will generate candidate images. Local feature matching and geometric verification will assess agreement using match counts, inlier ratios, and matched-region coverage. These thresholded signals would inform the final decision of *comparable, same building but different view, no match, or uncertain*.

**Evidence-backed responses.** A fixed vision-language model will receive the question and verified evidence relevant to the workflow. Its structured response will record the outcome, claims, supporting image, and limitations before producing readable text. The system will return “not found” when no identity is established and “insufficient evidence” when a relevant building is found but the requested answer cannot be verified. Missing records may trigger a bounded web lookup; unresolved evidence will lead to a clarification request or refusal.

**Evaluation and logging.** Every request will log its plan, executed steps, retrieved candidates, verification results, sources, response time, and estimated cost. A fixed test set will include all three workflows and cases requiring refusal.

**Data, evaluation, and tests**

We will use historical and recent photographs from the Georgia Tech Image Database, along with new campus photographs taken by the team.10 The evaluation will test whether MapEvolve retrieves useful evidence, answers supported questions correctly, and declines questions that the available evidence cannot resolve. We will construct approximately 60 cases: 18 building-information questions, 24 change questions, and 18 two-building proximity questions. Each case will contain a query image, question, reference answer, and supporting image IDs or records. These reference labels will be available to evaluators, not supplied to the system. Approximately one-fifth of the cases will require refusal. These will distinguish no valid match, such as a non-campus photograph, from insufficient evidence, such as an incompatible viewpoint, an obscured detail, or a missing record. Change cases will also distinguish a visible change from no visible change in the regions that can be compared. We will reserve approximately 20 cases for development and 40 for held-out evaluation, preserving the question-type proportions where possible. Closely related photographs and questions will remain in the same split to reduce leakage. Thresholds and prompts will be frozen before held-out testing. Another teammate will review each case; no one will approve their own photo pair. Reference coordinates will use a consistent, documented point for each building and the Student Center (our reference landmark), with the distance definition fixed before evaluation.

**Retrieval and verification.** For cases with valid corpus matches, we will report the proportion whose correct building appears in the top one and top three results. For change questions, we will separately measure whether these results include a usable historical comparison, since finding the correct building does not necessarily provide a comparable view. Verification will be scored against reviewed pair labels using precision and recall for comparable pairs, the rate of incompatible pairs accepted, and the rate of comparable pairs rejected. We will report answer coverage (the proportion of requests receiving a substantive answer) alongside accuracy among answered requests, so that refusing everything cannot appear successful.

**Question-level performance**. A short scoring guide will define acceptable answers and partial credit. A second rater will independently score at least 20% of held-out cases, distributed across question types and refusal cases; we will report agreement before resolving disagreements. Across all questions, we will measure correct refusal, unnecessary refusal on answerable cases, and unsupported answers on cases requiring refusal. Refusal reasons must match the reference limitation. Failures will be assigned a primary cause (retrieval, visual verification, record lookup, distance calculation, or answer generation) with secondary causes noted when necessary.

**Efficiency and demonstration.** The configurations defined in the proposed-work section will be evaluated on the same held-out cases. The fixed and question-specific workflows will use the same models and evidence sources, with cache conditions controlled. We will report answer quality, median and 95th-percentile latency, and estimated cost per request, both overall and by question type. The final demonstration will show all three question types and one justified refusal, with supporting evidence and reference outcomes visible.

**Plan of action**

| Week | Samil | Shaktik | Alejandro | Kausar | Milestone |
| :---- | :---- | :---- | :---- | :---- | :---- |
| 7 | Set up pair-verification pipeline | Set up image-search pipeline and initial corpus | Bring the draft evaluation framework and the first batch of test questions to the meeting.  | Lead professor meeting; finalize scope & components | Professor meeting; finalize scope and responsibilities |
| 8 | Implement initial photo-pair checks using local matching and return reviewed pair decisions | Implement CLIP/image retrieval and return ranked candidates | Build the first 20 to 25 cases, mostly building and distance cases. Assign the development/test split | Integrate retrieval, pair verification, answer generation; initial examples | Finalize initial end-to-end examples and cases |
| 9 | Improve pair checks using viewpoint/geometric verification | Improve retrieval using building metadata and records | Run the first internal test on all three question types. Start human rating and produce a first failure table | Complete three workflows, coordinate lookup, refusal handling | Begin internal testing for all query types |
| 10 | Analyze pair check errors and improve verification rules | Analyze retrieval errors and improve search ranking | Review results with the team, assign each failure to its corresponding component | Incorporate upgrades, compare fixed & specific workflows | Review testing results and begin improvements |
| 11 | Fill gaps in image pair examples and integrate verification with retrieval  | Fill important gaps in image search/database | Complete the 60-case test case and run the second check on 20% of labels. Rerun tests.  | Deliver stable, presentation-ready demo incorporating key artifacts | Finalize demonstration and workshop presentation |
| 12 | Finalize pair verification results and reviewed source pair information | Finalize retrieval results and source information | Freeze the test split and run the final evaluation. Build the main results table  | Freeze system settings, support final eval, finalize results | Finalize workshop-ready demonstration |
| 13 | Fix pair verification issues  | Fix search/retrieval issues found during testing | Rehearse the workshop and build a demo check | Lead workshop rehearsal, fix blocking issues | Workshop Presentation |
| 14 | Verify pair labels, comparable views, and evidence used in demo | Verify image records, building data, and sources | Incorporate workshop feedback. Rerun regressions for fixes that affect the demo. Freeze demo examples | Incorporate workshop feedback, coordinate regression checks | Finalize Professor demonstration |
| 15 | Final check of pair-verification components | Final check of retrieval/database components | Final run-through on the demo device, with a backup path if a service is down | Package deliverables, demonstrate workflows, document limitations | Professor demonstration |

**Bibliography**

\[1\] Bae, Soonmin, Aseem Agarwala, and Frédo Durand. "Computational rephotography." ACM Trans. Graph. 29.3 (2010): 24-1.

\[2\] Arandjelovic, Relja, et al. "NetVLAD: CNN architecture for weakly supervised place recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016\.

\[3\] Sattler, Torsten, et al. "Benchmarking 6dof outdoor visual localization in changing conditions." 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition. IEEE, 2018\.

\[4\] Sarlin, Paul-Edouard, et al. "Superglue: Learning feature matching with graph neural networks." 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2020\.

\[5\] Park, Dong Huk, Trevor Darrell, and Anna Rohrbach. "Robust change captioning." 2019 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2019\.

\[6\] Lewis, Patrick, et al. "Retrieval-augmented generation for knowledge-intensive nlp tasks." Advances in neural information processing systems 33 (2020): 9459-9474.

\[7\] Wu, Yin, et al. "Visual-rag: Benchmarking text-to-image retrieval augmented generation for visual knowledge intensive queries." Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2026\.

\[8\] Gupta, Tanmay, and Aniruddha Kembhavi. "Visual programming: Compositional visual reasoning without training." 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2023\.

\[9\] Patherya, Kausar. “POLARIS: Finding the Same Landmark in a Different Season.”, 5 Dec. 2025, https\://kausarpatherya.com/2025/12/05/polaris-cross-season-landmarks/index.html.

\[10\] “Georgia Tech Image Database.”, https\://gatech.netx.net/portals/georgia-tech-image-database/. Accessed 2 Oct. 2026\.
