# 모아뛰기 기술 문서

## 0. 초기 버전 한계와 오프라인 후처리 필수성

### 0.1 초기 버전의 구조적 문제점

- 검출이 단일 규칙에 과도하게 의존
  - `left_hip_y` 로컬 최소점 + 고정 임계치(`diff >= 0.003`) 방식이라 촬영 조건/피사체/리듬 변화에 취약합니다.
- 시간적 제약과 상태 전이가 없음
  - 최소 점프 간격, 시작 lockout, active/inactive 상태 관리가 없어 경계 구간/노이즈에서 과검출이 쉽게 발생합니다.
- rope evidence가 최종 카운트에 구조적으로 결합되지 않음
  - rope flag를 표시하지만 점프 이벤트 확정 로직 자체를 강하게 제어하지 못합니다.

- 성능
  - 매 프레임마다 과거 전체 시계열을 다시 스캔하는 형태라 영상 길이가 길어질수록 비효율이 커집니다.


### 0.2 오프라인 후처리가  필요한 이유

온라인 1-pass 검출만으로는 모든 프레임에서 안정적인 카운트를 보장하기 어렵습니다. 실제 영상에서는 다음이 반복됩니다.

- 순간 pose 누락/오검출
- 모션 블러/가림으로 인한 최소점 약화
- 시작/종료 경계의 비정상 리듬
- 구간별 cadence 변화

이 때문에 최종 카운트는 전체 시계열을 본 오프라인 재정리가 필요합니다.

1. 큰 gap 구간 누락 복원 (`recover_missing_events_in_large_gaps`)
2. 단일 미스 보간 (`recover_high_cadence_single_miss_gaps`)
3. 시작/끝 작은 세그먼트 제거 (`prune_small_edge_segments`)
4. dual ratio 기반 세그먼트 프루닝 (`prune_low_dual_ratio_segments`)
5. 경계 저신뢰 이벤트 제거 (`prune_low_confidence_boundary_events`)
6. 라벨 평가 시 tolerance/strict/adjusted 지표로 정량 검증 (`evaluate_detected_events`)

## 1. 전체 처리 파이프라인

영상 1개(job) 기준 처리 순서는 아래와 같습니다.

1. 입력 job 생성
   - 모드별로 대상 영상/라벨 목록 구성
2. 프레임 루프
   - MediaPipe Pose 랜드마크 추출
   - 손목 ROI 기반 rope evidence 추출
   - 온라인 카운트(오버레이용 임시 카운트)
   - 트래킹 영상 저장(`*_tracked.mp4`)
3. 오프라인 재검출(`detect_jump_events_offline`)
   - 전체 시계열 기반으로 점프 이벤트 재구성
   - gap/interp/prune 후처리 적용
4. 결과 정리
   - 라벨 없으면: 검출 CSV + 요약
   - 라벨 있으면: 이벤트 매칭, 성능 지표, 비교 CSV, 요약
5. 요약 CSV 저장
   - 모드별 summary 파일(`all_videos_summary.csv`, `video_count_summary.csv`, `realtime_count_summary.csv`)

중요:

- 최종 카운트는 3단계 오프라인 재검출 결과를 사용합니다.
- 라벨은 `labeled` 모드에서 평가 전용이며, 검출 카운트 자체를 보정하지 않습니다.

## 2. 프레임 루프 핵심 로직

### 2.1 Pose 기반 시계열 생성

Pose 검출 성공 프레임에 대해 다음 시계열을 누적합니다.

- `hip_center_y = (left_hip_y + right_hip_y)/2`
- 발목 좌표(`left_foot_x/y`, `right_foot_x/y`)
- 프레임 인덱스/타임스탬프

타임스탬프는 `resolve_frame_timestamp_ms`로 보정합니다.

- `CAP_PROP_POS_MSEC` 이상치/역행 방지
- FPS 기반 fallback 시간 사용

### 2.2 Rope evidence 추출

rope evidence는 손목 주변 ROI + KNN foreground로 계산합니다.

1. 좌/우 손목 중심 ROI를 생성
2. `cv2.createBackgroundSubtractorKNN` 마스크에서 contour 추출
3. contour area가 임계값 이상일 때 ROI hit 판단
4. `rope_detected_this_frame`(좌/우 중 하나), `rope_dual_detected_this_frame`(양손 동시) 생성

이 bool 시계열은 이후 검출 게이트(활성/진입/경계 프루닝)에 사용됩니다.

### 2.3 온라인 카운트(오버레이용)

프레임 루프 안에서 로컬 최소점 기반 온라인 카운트를 수행합니다.

- 적응형 threshold + prominence + 최소 점프 간격
- rope ratio 게이트
- 활성 상태 전이(`inactive -> active`) 후 카운트

온라인 카운트는 즉시 화면/영상 오버레이를 위해 사용되며, 종료 후 오프라인 재검출로 최종 이벤트를 다시 확정합니다.

## 3. 오프라인 후처리 순서

`detect_jump_events_offline` 내부 적용 순서:

1. `recover_missing_events_in_large_gaps` (`ENABLE_GAP_RECOVERY`)
2. `recover_high_cadence_single_miss_gaps` (`ENABLE_HIGH_CADENCE_GAP_INTERP`)
3. `recover_high_cadence_single_miss_gaps` long-run 설정 (`ENABLE_LONG_RUN_GAP_INTERP`)
4. `prune_small_edge_segments` (`ENABLE_EDGE_SEGMENT_PRUNE`)
5. `prune_low_dual_ratio_segments` (`ENABLE_SEGMENT_DUAL_PRUNE`)
6. `prune_low_confidence_boundary_events` (`ENABLE_BOUNDARY_CONF_PRUNE`)

### 3.1 Large-gap recovery

- 이벤트 간격의 median을 기준으로 큰 gap 구간 탐지
- 후보 강도/rope 조건/중복 간격 조건을 만족하는 minima를 삽입
- gap별 삽입 개수 상한 적용

### 3.2 High-cadence / Long-run 단일 미스 보간

- 주변 interval CV가 낮은 안정 구간에서만 동작
- gap ratio가 거의 정수배이고 `missing_count == 1`일 때 중간 이벤트 1개 삽입
- 좌표/시간은 양끝 이벤트 선형 보간

### 3.3 Edge segment prune

- 큰 시간 간격으로 세그먼트 분할
- 시작/끝의 짧은 세그먼트를 제거(메인 세그먼트가 충분히 긴 경우만)

### 3.4 Segment dual prune (zero-collapse guard 포함)

- 세그먼트별 `rope_dual_ratio` 중앙값 + cadence 안정성(CV) 기반으로 저신뢰 세그먼트를 제거
- 단, 과도한 규제로 모든 세그먼트가 제거되는 경우를 방지하기 위해 fallback 적용
  - cadence 일관성이 있는 세그먼트를 우선 복구
  - dual 결손이 큰 경우 저신뢰 이벤트를 소량 trim 하여 과탐 완화
  - 결과적으로 `offline_refined=0`으로 붕괴하는 케이스를 방지

### 3.5 Boundary confidence prune

- 헤드/테일 이벤트의 rope ratio/dual ratio가 절대적으로 낮거나,
  인접 reference 프로파일 대비 상대적으로 낮으면 제거
- 단, 옆 이벤트가 충분히 강한 경우에만 제거

## 4. 라벨 평가 로직(`labeled`)

### 4.1 매칭 tolerance

`tolerance_ms = max(JR_MATCH_TOLERANCE_MS, frames->ms)` 후 상한(`JR_MATCH_TOLERANCE_MAX_MS`) 적용

### 4.2 이벤트 매칭

정렬된 detected/label를 two-pointer 방식으로 1:1 시간 매칭:

- tolerance 이내면 `matched`
- detected가 너무 이르면 `extra_detected`
- label이 너무 이르면 `missed_labels`

### 4.3 strict / adjusted 지표

`split_gap_candidate_events`:

- 라벨 내부의 큰 공백(중앙 근처)에서 생긴 extra를 `label_gap_candidate`로 분리
- `JR_STRICT_IGNORE_GAP_CANDIDATES=true`이면 strict extra에서 gap candidate를 제외

최종 산출:

- `strict_precision/recall/f1/accuracy`
- `adjusted_precision/recall/f1/accuracy`
- `mean_abs_time_error_ms`
- 좌/우 발 위치 오차 평균(px)

## 5. 핵심 기본 설정(현재 코드 기본값)

### 5.1 검출

- `JR_THRESHOLD_GAIN=0.60`
- `JR_MINIMA_LAG_FRAMES=4`
- `JR_PROMINENCE_WINDOW=6`
- `JR_MIN_JUMP_GAP_SECONDS=0.17`
- `JR_MIN_STRENGTH_RATIO=1.0`
- `JR_STARTUP_LOCKOUT_SECONDS=0.8`
- `JR_LANDING_OFFSET_MS=100`
- `JR_EVENT_TIME_BIAS_MS=100`

### 5.2 상태/rope

- `JR_ACTIVE_ENTER_WINDOW_SECONDS=0.8`
- `JR_ACTIVE_ENTER_MIN_EVENTS=2`
- `JR_ACTIVE_ENTER_MAX_GAP_SECONDS=0.55`
- `JR_ACTIVE_ENTER_CADENCE_MAX_CV=0.55`
- `JR_ACTIVE_EXIT_IDLE_SECONDS=0.8`
- `JR_ROPE_ACTIVE_WINDOW_SECONDS=0.6`
- `JR_ROPE_ACTIVE_MIN_RATIO=0.00`
- `JR_ROPE_ENTRY_MIN_RATIO=0.00`
- `JR_ROPE_ENTRY_DUAL_MIN_RATIO=0.00`
- `JR_ROPE_EXIT_IDLE_SECONDS=0.8`

### 5.3 후처리 on/off

- `JR_ENABLE_GAP_RECOVERY=true`
- `JR_ENABLE_HIGH_CADENCE_GAP_INTERP=false`
- `JR_ENABLE_LONG_RUN_GAP_INTERP=true`
- `JR_ENABLE_EDGE_SEGMENT_PRUNE=true`
- `JR_ENABLE_SEGMENT_DUAL_PRUNE=true`
- `JR_SEGMENT_DUAL_PRUNE_MIN_MEDIAN=0.50`
- `JR_ENABLE_SEGMENT_CADENCE_PRUNE=true`
- `JR_SHORT_SEGMENT_CADENCE_MAX_CV=0.20`
- `JR_ENABLE_BOUNDARY_CONF_PRUNE=true`

### 5.4 평가/출력

- `JR_MATCH_TOLERANCE_MS=120`
- `JR_MATCH_TOLERANCE_FRAMES=5.0`
- `JR_MATCH_TOLERANCE_MAX_MS=180`
- `JR_LABEL_WINDOW_PADDING_MS=150`
- `JR_STRICT_IGNORE_GAP_CANDIDATES=true`
- `JR_ENABLE_OVERLAY_REFRESH=true`
- `JR_LIVE_OVERLAY_REFINED_COUNT=false` (기본 live 오버레이를 raw 카운트로 통일)
- `JR_OVERLAY_END_WAIT_ENABLED=true` (영상 종료 후 overlay 값 안정화 대기)
- `JR_OVERLAY_END_WAIT_SECONDS=1.5` (기본 대기 시간)
- `JR_OVERLAY_END_WAIT_MAX_SECONDS=8.0` (최대 대기 상한)

## 6. 최신 검증 결과 (2026-03-03)

### 6.1 최신 realtime 세션 재검증

대상 세션:

- `basic_jump/output/realtime_demo_logs/realtime_cam0_20260302_224101`

재현 방식:

- `main_video.py --video-path .../raw.mp4`
- `main_video.py --video-path .../tracked.mp4`

결과:

- raw 재생: `detected_count=20`
- tracked 재생: `detected_count=20`

### 6.2 labeled full(01~10) 결과

검증 방식:

- `main_label.py --target-stem 01..10` 개별 실행
- strict/adjusted 지표 수집

결과 요약:

- 전 stem에서 `strict_f1=1.0`
- 전 stem에서 `adjusted_f1=1.0`
- `full_*` 지표는 일부 stem에서 1.0 미만(라벨 윈도우 바깥 extra 영향)이며 strict/adjusted 품질과 별도 관리
