# AGENTS.md

## Project Overview
This project aims to build a grouped-jump (모아뛰기) counter using:

- Kinovea labels in `/label`
- Raw videos in `/video`
- MediaPipe landmarks extracted from the videos

The current phase is NOT implementation-heavy.
The primary goal is **deep analysis and strategy definition**.

---

## Core Objective (Current Phase)
Before building any counter logic, you must:

1. Analyze Kinovea label structure and semantics
2. Analyze raw video timing properties
3. Verify timestamp alignment between labels and videos
4. Define what "1 count" means operationally
5. Identify observable MediaPipe signals
6. Propose a real-time-compatible counting strategy

Do NOT skip analysis.

---

## Non-Negotiable Constraints

### 1. Timestamp Integrity (Highest Priority)
- NEVER assume label timestamps are correct
- ALWAYS verify alignment with raw video
- Consider:
  - FPS mismatch
  - VFR vs CFR
  - export offsets
  - frame/time conversion errors

### 2. UX Constraints (Critical)
The system MUST:
- Support immediate or near-real-time feedback
- Avoid full-video offline processing as a requirement
- Avoid long post-processing delays

DO NOT propose:
- batch-only pipelines
- full video post-hoc correction as required step
- future-frame dependent logic unless justified

### 3. No Assumptions Without Evidence
- Do NOT guess label meaning
- Do NOT assume frame alignment
- Do NOT assume file pairing

Always verify using actual data.

---

## Data Validation Requirements

You MUST explicitly verify:

- label ↔ video file mapping
- timestamp unit and precision
- whether timestamps are:
  - absolute time
  - frame index
  - adjusted timeline
- video properties:
  - FPS
  - duration
  - CFR/VFR behavior
- whether labels correspond to:
  - takeoff
  - landing
  - mid-air
  - accepted count moment

---

## Analysis Priorities

1. Label schema understanding
2. Video metadata inspection
3. Label-video alignment verification
4. Deep inspection of several labeled events
5. Identification of robust landmarks/signals
6. Definition of "1 count"
7. Strategy design (real-time compatible)
8. Risk identification

---

## Implementation Philosophy (Future)

Any proposed system should:

- Work in real-time or near real-time
- Be causal or minimally look-ahead
- Avoid dependence on full-sequence hindsight
- Be robust to noisy beginner motion
- Be explainable from observed signals

---

## Required Outputs

You MUST produce:

1. Label schema documentation
2. Video timing analysis
3. Alignment validation report
4. Operational definition of "1 count"
5. Candidate landmark/signal list
6. Real-time strategy proposal
7. Risk & uncertainty list

---

## Completion Criteria

This phase is complete ONLY if:

- Label semantics are clearly defined
- Timestamp alignment is verified or problematized
- Several samples are deeply analyzed
- A viable real-time strategy is proposed
- All uncertainties are explicitly documented