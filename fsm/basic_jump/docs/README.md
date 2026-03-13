# Basic Jump Detector Tech Docs

## 1) Prerequisites

이 문서를 읽기 전에 아래 개념을 먼저 잡아두면 전체 구조를 이해하기 쉽습니다.

- `raw candidate`
  - 프레임 루프에서 생성된 원시 jump event입니다.
  - 아직 최종 count가 아닙니다.
- `confirmed count`
  - `all-in-one` 선택기를 통과한 이벤트만으로 만든 최종 count입니다.
  - 라이브 오버레이와 저장 결과가 이 count를 공유합니다.
- `cadence`
  - 연속 jump 간 시간 간격의 규칙성입니다.
  - CV가 낮을수록 안정적인 점프열로 봅니다.
- `rope ratio` / `rope dual ratio`
  - 손목 ROI에서 rope evidence가 얼마나 안정적으로 관측되는지를 나타내는 비율입니다.
  - `dual`은 좌/우가 동시에 보인 정도입니다.
- `bilateral / symmetry`
  - 양발 lift가 함께 보이거나 좌우 발 움직임이 대칭적인지에 대한 증거입니다.
- `raw commit lag` / `positive commit lag`
  - `raw_commit_lag_ms = overlay_count_reached_ts - final_event_ts`
  - display advance와 event timestamp bias 때문에 raw lag는 음수가 될 수 있습니다.
  - UX 평가는 `positive_commit_lag_ms = max(0, raw_commit_lag_ms)`로 해석합니다.
- `strict / adjusted / full`
  - `strict`: label window 내부에서 직접 비교한 지표
  - `adjusted`: label gap candidate를 제외한 지표
  - `full`: label window 밖 confirmed extra까지 포함한 지표
- `label_boundary_candidate`
  - 첫/마지막 label 바로 바깥에 있지만 cadence상 같은 점프열로 이어지는 경계 이벤트입니다.
  - runtime 오탐이 아니라 평가 보고용 분류입니다.
- `label timestamp conversion`
  - Kinovea 고주파 tick 라벨은 파일 timebase를 이용해 ms로 변환한 뒤 비교합니다.

## 2) What

이 시스템은 줄넘기 jump count를 `label / video / realtime`에서 같은 엔진으로 계산하기 위한 detector입니다.

현재 운영 기준의 핵심은 아래 두 가지입니다.

- 기본 엔진은 `all-in-one` 온라인 확정 엔진입니다.
- 전체 영상을 다시 스캔하는 offline 재검출 경로는 운영 기준에서 사용하지 않습니다.

즉 현재 구조의 목적은 단순히 jump event를 많이 잡는 것이 아니라,
`라이브에서 보인 count`와 `최종 저장 count`가 같은 의미를 갖도록 만드는 것입니다.

## 3) Why

이 구조가 필요한 이유는 UX 문제 때문입니다.

기존 문제:

- 빠른 임시 count와 종료 후 최종 count가 달라질 수 있었습니다.
- no-jump나 짧은 오탐 세션에서 숫자가 먼저 올라갔다가 사라질 수 있었습니다.
- 사용자는 라이브에서 본 숫자를 신뢰하기 어렵습니다.

현재 설계 목표:

- `final_count = online_confirmed_count`
- 종료 후 과거 이벤트를 다시 분류해서 count를 바꾸지 않음
- count는 monotonic하게 증가하고 rollback하지 않음
- 확정 지연은 전체 영상 길이가 아니라 세션 초반 prefix와 로컬 gap에만 의존함

## 4) Where

같은 엔진이 세 경로에 공통 적용됩니다.

- `label`
  - 영상과 라벨이 모두 있을 때 count와 평가를 함께 수행합니다.
- `video`
  - 영상만 있을 때 count를 수행하고, 라벨이 있으면 같은 방식으로 비교합니다.
- `realtime`
  - 카메라 입력에서 같은 count 엔진을 사용합니다.

세 경로는 입력 소스와 평가 유무만 다르고, 카운팅 원리는 같습니다.

## 5) Who

이 시스템에서 실제 역할을 나누는 구성요소는 아래와 같습니다.

- `frame loop`
  - pose, rope evidence, foot motion을 프레임 단위로 읽습니다.
- `raw candidate detector`
  - minima, motion, rope, gap 조건을 통과한 프레임을 raw candidate로 만듭니다.
- `all-in-one session confirmer`
  - raw candidate를 바로 count하지 않고 세션 단위로 재구성합니다.
- `bounded recovery`
  - 세션 시작부와 단일 gap 누락만 온라인으로 복원합니다.
- `overlay renderer`
  - 현재 confirmed count만 화면 숫자로 표시합니다.
- `evaluator`
  - label과 비교해 strict / adjusted / full 지표를 계산합니다.

## 6) When

각 단계는 아래 시점에 동작합니다.

- `raw candidate` 생성 시점:
  - pose와 하체 landmark가 충분히 신뢰 가능하고
  - minima와 motion signature가 맞고
  - rope evidence와 jump gap 조건을 통과할 때 생성됩니다.
- `confirmed count` 증가 시점:
  - raw candidate가 세션으로 묶이고
  - 세그먼트 시작점이 유효하다고 판단된 뒤 증가합니다.
- `entry backfill` 적용 시점:
  - 세션이 이미 유효하다고 판단된 뒤, 바로 앞 1~2개 raw event가 cadence상 자연스럽게 이어질 때만 적용합니다.
- `gap recovery` 적용 시점:
  - 확정 세션 내부에서 단일 cadence 누락이 ratio상 분명할 때만 적용합니다.
- `stable` 시작을 여는 시점:
  - 현재 기본값은 `JR_ALL_IN_ONE_STABLE_ENTRY_MIN_EVENTS=4`입니다.
  - 이 값은 `11`의 초기 지연을 `1200ms -> 333ms`로 줄이기 위해 `5 -> 4`로 조정됐습니다.
- evaluation 보정 적용 시점:
  - `label_boundary_candidate`와 `label_gap_candidate`는 runtime count를 바꾸지 않고 보고 단계에서만 분리됩니다.

## 7) How

### 7.1 Raw Candidate Generation

프레임 루프는 아래 순서로 원시 후보를 만듭니다.

1. pose 추출
2. 하체 신뢰도 검사
3. 손목 ROI 기반 rope evidence 추출
4. hip/foot 시계열에서 로컬 minima 후보 생성
5. motion / gap / rope 게이트 통과 시 raw candidate 생성

하체 신뢰도 기본 조건:

- 양쪽 hip가 모두 신뢰 가능해야 함
- 좌/우 하체(무릎/발목/발끝)에서 각 측 최소 1개 이상 신뢰 가능해야 함

핵심 게이트:

- jump gap gate
- motion signature gate
- rope ratio / rope dual ratio gate
- active/inactive state gate
- anti-walk gate
- entry bootstrap gate

### 7.2 All-In-One Session Confirmation

raw candidate는 바로 최종 count가 되지 않습니다.
`_select_all_in_one_events(detected_jump_events, ...)`가 세션 단위로 유효 jump만 남깁니다.

처리 순서:

1. weak non-bilateral prune
2. gap 기반 세그먼트 분리
3. 세그먼트 시작점 판정
4. bounded entry backfill
5. bounded gap recovery
6. count 재부여

세그먼트 시작점 판정 유형:

- `strong_high_dual`
  - 초반 cadence가 안정적이고 dual rope 증거가 강한 세션
- `strong_low_dual`
  - dual이 약해도 rope support와 strong-positive가 충분한 세션
- `stable`
  - 더 긴 prefix 안정성으로 시작을 인정하는 세션

### 7.3 Bounded Online Recovery

offline 재스캔 대신 작은 로컬 복원만 허용합니다.

- `entry backfill`
  - 확정 세션 바로 앞 1~2개 실제 raw event를 다시 포함
  - 목적: 세션 시작 직전 실제 jump가 늦게 확정되는 문제 복구
- `gap recovery`
  - 확정 세션 내부의 단일 cadence 누락만 온라인 보간
  - 핵심 기본값: `JR_ALL_IN_ONE_GAP_FILL_MIN_RATIO=1.50`

이 제한 때문에 현재 구조는 future 전체를 다시 뒤집는 offline 후처리와 다릅니다.

### 7.4 Overlay and Final Count

현재 운영 UX의 핵심은 여기입니다.

- 화면에는 현재 count 숫자만 표시합니다.
- pose landmarks는 계속 표시합니다.
- 라이브 오버레이와 최종 저장 count는 같은 all-in-one 결과를 사용합니다.
- 즉 종료 후 count source가 다른 규칙으로 바뀌지 않습니다.

### 7.5 Evaluation

평가는 runtime과 별도입니다.

- `strict`
  - label window 내부 직접 비교
- `adjusted`
  - label gap candidate 제외
- `full`
  - confirmed outside-window extra 포함

경계 이벤트 해석:

- `01`, `02`, `03`, `06`에는 총 `7`개의 경계 이벤트가 있습니다.
- 이 항목들은 `outside_label_window`가 아니라 `label_boundary_candidate`로 분류됩니다.
- 따라서 현재 `full=1.000`은 runtime count 개선이 아니라 평가 정의 정리 결과를 포함합니다.

## 8) Current Status

`2026-03-12 00:55 KST` 기준:

정확도:

- `01~11 strict_f1_mean=1.000`
- `strict_f1_min=1.000`
- `adjusted_f1_mean=1.000`
- `full_strict_f1_mean=1.000`
- `full_adjusted_f1_mean=1.000`
- no-jump `detected_count=0`

대표 UX:

- `07`: `final_live_delta=0`, `monotonic=true`, `max_positive_commit_lag_ms=0`
- `09`: `final_live_delta=0`, `monotonic=true`, `max_positive_commit_lag_ms=500`
- `11`: `final_live_delta=0`, `monotonic=true`, `max_positive_commit_lag_ms=333`
- no-jump: `overlay_max_count=0`, `final_count=0`

요약:

- representative probe 기준으로 `e2e <= 1000ms`와 `live=final`은 현재 충족합니다.
- 정확도 1.000과 UX gate는 모두 닫힌 상태입니다.
- 남은 일은 엔진 원리 변경이 아니라, 더 많은 세션 유형에 대해 같은 UX 수치를 지속 추적하는 것입니다.

참고 산출물:

- `output/realtime_engine_summary.csv`
- `output/ux_probe_summary.csv`
