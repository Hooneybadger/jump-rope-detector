# Basic Jump Detector Tech Docs

## 1) 범위

- 이 프로젝트는 **단일 엔진**으로 동작합니다.
- `label/video/realtime`는 입력 소스와 평가 유무만 다르고, 카운팅 엔진은 동일합니다.

## 2) 실행 진입점

- `main_label.py`
- `main_video.py`
- `main_realtime.py`

세 진입점 모두 내부적으로 동일한 `run_pipeline(...)`을 호출합니다.

## 3) 파이프라인 구조

### 3.1 입력 Job 구성

- `label`: 영상 + 라벨이 둘 다 있는 스템만 처리 (`require_label=True`)
- `video`: 영상만 처리 (라벨이 있으면 자동 비교, 없으면 카운트만 출력)
- `realtime`: 카메라 입력 처리 (`use_realtime=True`)

### 3.2 프레임 온라인 검출

프레임 루프에서 다음을 수행합니다.

- Pose 추출 + 하체 신뢰도 검사
- 손목 ROI 기반 rope evidence 추출
- hip/foot 시계열에서 로컬 minima 후보 생성
- 후보 게이트 통과 시 이벤트 생성

하체 신뢰도 기본 규칙:

- 양쪽 hip가 모두 신뢰 가능해야 함
- 좌/우 하체(무릎/발목/발끝)에서 각 측 최소 1개 이상 신뢰 가능해야 함

핵심 온라인 게이트:

- 점프 간격 게이트 (`JR_MIN_JUMP_GAP_SECONDS`)
- 모션 시그니처 게이트
- rope ratio / rope dual ratio 게이트
- active/inactive 상태 전이 게이트
- anti-walk 게이트 (`JR_STRICT_ACTIVE_ANTI_WALK_ENABLED`)
- entry bootstrap 게이트 (`JR_STRICT_ENTRY_NO_FLAG_BOOTSTRAP_ENABLED`)

### 3.3 Fixed-Lag 확정과 스트리밍 오버레이

- 최종 카운트의 입력은 `fixed_lag_confirmed_events`입니다.
- 기본값:
  - `JR_FIXED_LAG_SECONDS=1.0`
  - `JR_FIXED_LAG_FINALIZE_WAIT_SECONDS=2.0`
- 기본 오버레이는 fixed-lag count를 그대로 쓰지 않고,
  `fixed_lag_confirmed_events`에 **동일한 post 규칙을 스트리밍 재적용한 count**를 사용합니다.
- 이때 다시 보는 대상은 **전체 영상 프레임/원신호가 아니라 이미 확정된 이벤트 리스트**입니다.
- 즉 예전 offline 재검출처럼 전체 영상을 다시 스캔하지 않고,
  현재까지 누적된 `fixed_lag_confirmed_events`만 매 업데이트 시 재정리합니다.
- 제어 스위치:
  - `JR_LIVE_OVERLAY_FIXED_LAG_COUNT=true`
  - `JR_LIVE_OVERLAY_STREAM_POSTPROCESS=true`

의도:

- 라이브 숫자를 종료 후 확정치에 더 가깝게 맞춘다.
- no-jump 짧은 스파이크를 라이브 오버레이 단계에서 바로 숨긴다.

### 3.4 세션 후처리 (Post)

최종 이벤트는 아래 순서로 정리됩니다.

1. weak non-bilateral prune
2. session filter
3. session gap fill
4. session head fill
5. session median strength prune
6. run-level min events prune

session filter 상세:

- 세그먼트 분리: `JR_STRICT_SESSION_SPLIT_GAP_SECONDS`
- 최소 이벤트 수: `JR_SESSION_MIN_EVENTS`
- cadence CV 상한: `JR_STRICT_SESSION_MAX_CADENCE_CV`
- abs foot prominence 하한: `JR_STRICT_SESSION_MIN_ABS_FOOT_PROMINENCE`
- 짧은 스파이크 브리지: `JR_STRICT_SESSION_SHORT_BRIDGE_*`

gap/head fill 상세:

- gap fill: `JR_STRICT_SESSION_GAP_FILL_*`
- head fill: `JR_STRICT_SESSION_HEAD_FILL_*`

### 3.5 오버레이 정책

- 화면에는 **현재 카운트 숫자만** 표시합니다.
- pose landmarks는 영상 위에 계속 표시합니다.
- 라벨 수, delta, 기타 디버그 텍스트는 표시하지 않습니다.
- `stream_post`가 켜져 있으면 오버레이 숫자는 다음 규칙을 이미 반영한 값입니다.
  - weak non-bilateral prune
  - session filter
  - session gap fill
  - session head fill
  - session median strength prune
  - run-level min events prune
- 필요 시 `JR_ENABLE_OVERLAY_REFRESH=true`로 post 단계에서 카운트 오버레이 재기록을 수행합니다.

## 4) 평가 방식

### 4.1 label 모드

- strict/adjusted/full 지표를 계산합니다.
- 주요 지표:
  - `strict_f1`
  - `missed_labels`
  - `strict_extra_detected`

### 4.2 video / realtime 모드

- 라벨이 없으면 `detected_count` 중심 결과를 출력합니다.
- summary의 F1 계열은 `NaN`이 정상입니다.

### 4.3 라벨 타임스탬프 보정

- Kinovea 고주파 tick 포맷 라벨은 파일 timebase를 이용해 ms로 자동 변환합니다.
- 비교 tolerance는 프레임 기반/고정 ms를 함께 고려해 계산합니다.

## 5) 용어

- `short spike`: 포즈 흔들림 등으로 생기는 짧은 오탐 이벤트 묶음
- `causal`: 미래 프레임을 보지 않는 온라인 판정
- `fixed-lag`: 일정 지연 후 이벤트를 확정하는 방식
- `session gate`: 이벤트 묶음을 유효 세션으로 인정하는 조건
- `rope ratio`: 최근 윈도우에서 rope evidence가 관측된 비율
- `rope dual ratio`: 좌/우 ROI 동시 rope evidence 비율

## 6) 현재 기본값

- `JR_STRICT_GUARDS=true`
- `JR_STRICT_LOWER_BODY_VIS_MIN=0.15`
- `JR_STRICT_ACTIVE_ANTI_WALK_ENABLED=true`
- `JR_STRICT_WEAK_NON_BILATERAL_PRUNE_ENABLED=true`
- `JR_STRICT_WEAK_NON_BILATERAL_MAX_STRENGTH=2.0`
- `JR_SESSION_MIN_EVENTS=5`
- `JR_STRICT_SESSION_MIN_MEDIAN_STRENGTH_RATIO=6.0`
- `JR_STRICT_SESSION_FILTER_ENABLED=true`
- `JR_STRICT_SESSION_SPLIT_GAP_SECONDS=0.90`
- `JR_STRICT_SESSION_MAX_CADENCE_CV=0.30`
- `JR_STRICT_SESSION_MIN_ABS_FOOT_PROMINENCE=0.0005`
- `JR_STRICT_SESSION_SHORT_BRIDGE_ENABLED=true`
- `JR_STRICT_SESSION_SHORT_BRIDGE_MIN_EVENTS=2`
- `JR_STRICT_SESSION_SHORT_BRIDGE_MAX_EVENTS=4`
- `JR_STRICT_SESSION_SHORT_BRIDGE_MAX_GAP_SECONDS=1.40`
- `JR_STRICT_SESSION_SHORT_BRIDGE_CV_SCALE=1.20`
- `JR_STRICT_SESSION_GAP_FILL_ENABLED=true`
- `JR_STRICT_SESSION_GAP_FILL_MIN_EVENTS=12`
- `JR_STRICT_SESSION_GAP_FILL_MAX_CADENCE_CV=0.22`
- `JR_STRICT_SESSION_GAP_FILL_MIN_RATIO=1.80`
- `JR_STRICT_SESSION_GAP_FILL_MAX_RATIO=8.00`
- `JR_STRICT_SESSION_HEAD_FILL_ENABLED=true`
- `JR_STRICT_SESSION_HEAD_FILL_MIN_EVENTS=300`
- `JR_STRICT_SESSION_HEAD_FILL_INSERT_COUNT=2`
- `JR_STRICT_SESSION_HEAD_FILL_MAX_CADENCE_CV=0.10`
- `JR_FIXED_LAG_SECONDS=1.0`
- `JR_FIXED_LAG_FINALIZE_WAIT_SECONDS=2.0`
- `JR_LIVE_OVERLAY_FIXED_LAG_COUNT=true`
- `JR_LIVE_OVERLAY_STREAM_POSTPROCESS=true`

## 7) 명령어

### 7.1 라벨 전체 검증 (1~11)

```bash
env -u DISPLAY JR_ENABLE_OVERLAY_REFRESH=false python main_label.py
```

### 7.2 라벨 단일 스템 검증

```bash
env -u DISPLAY JR_ENABLE_OVERLAY_REFRESH=false python main_label.py --target-stem 07
```

### 7.3 비라벨 영상 카운트

```bash
env -u DISPLAY JR_ENABLE_OVERLAY_REFRESH=false python main_video.py --video-path <video_path>
```

### 7.4 realtime 시연 + 저장

```bash
python main_realtime.py --camera-index 0 --demo-log --demo-save-raw
```

- 결과 저장 위치: `output/realtime_demo_logs/<session_dir>/`
- 주요 산출물:
  - `tracked.mp4`
  - `raw.mp4` (`--demo-save-raw` 사용 시)
  - `frame_log.csv`
  - `<stem>_detected_events.csv`

### 7.5 summary CSV

- video/label 실행 요약: `output/realtime_engine_summary.csv`
- realtime 실행 요약: `output/realtime_count_summary.csv`

## 8) 최신 성능 평가

- label 01~11: `strict_f1=1.000`
- no-jump 샘플:
  - `output/realtime_demo_logs/demo_20260309_000620_realtime_cam0_20260309_000621/raw.mp4`
  - 결과: `detected_count=0`

## 9) UX 편차 실측 (라이브 vs 종료 후 최종 카운트)

### 9.1 요약

- 요지는 **오버레이를 "빠른 추정치"가 아니라 "보수적 확정치"에 더 가깝게 바꿨다**는 것입니다.
- 이전 fixed-lag 오버레이는 짧은 오탐 세션도 바로 숫자로 보여줘서, 사용자 입장에서는 의미 없는 숫자가 먼저 올라가고 종료 후 사라질 수 있었습니다.
- 현재 `stream_post` 오버레이는 같은 세션 규칙을 즉시 반영하므로, **보여준 숫자가 최종 숫자로 남을 가능성**을 높입니다.
- 대가로 **실제 점프 시작 직후 몇 카운트는 표시가 늦어질 수 있습니다.**

### 9.2 실측 근거

정량 기준:

- `count_delta_end = final_count - live_overlay_last_count`
- `first_event_time_gap_ms = first_live_count_ts - first_final_event_ts`

`01~11` 라벨셋에서 `first_pass -> final` 편차:

- 11개 중 8개 스템: 보정 0
- 보정 발생: `02(+2)`, `07(+3)`, `11(+4)`
- 평균 `0.818 count`, 중앙값 `0`, 최대 `+4` (`11`, `26 -> 22`)

realtime demo 로그(overlay_counter 있는 세션) 편차:

- 집계 4세션
- `count_delta_end`: min `0`, max `+7`, mean `2.0`, median `0.5`
- 예시:
  - `demo_20260309_000620...`: `47 -> 48` (`+1`)
  - `realtime_cam0_20260302_212705`: `17 -> 24` (`+7`)

### 9.3 이번 시도에서 확인한 개선 효과

- 11번 샘플:
  - 기존 fixed-lag 오버레이 마지막 값: `24`
  - 새 `stream_post` 오버레이 마지막 값: `22`
  - 최종 확정값: `22`
  - 첫 fixed-lag 확정 대비 첫 오버레이 표시는 약 `+2000 ms` 늦어짐
- no-jump 샘플(`demo_20260309_000620.../raw.mp4`):
  - 기존 fixed-lag/raw 내부 카운트는 끝까지 `45`
  - 새 `stream_post` 오버레이는 끝까지 `0`
  - 최종 확정값: `0`

즉 현재 시도는 UX 관점에서 다음 효과를 냈습니다.

- no-jump에서 의미 없는 숫자가 보이는 문제를 사실상 제거
- 실제 점프 세션에서는 라이브 마지막 숫자와 최종 숫자의 불일치를 축소
- 오버레이 숫자는 더 늦게 올라오지만, 올라온 뒤 다시 뒤집힐 가능성은 줄어듦
