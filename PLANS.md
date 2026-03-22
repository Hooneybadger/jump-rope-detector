# PLANS.md

## 목표

`/basic_jump/video` 테스트셋과 대응 Kinovea 라벨을 사용해,
**MediaPipe 기반 모아뛰기 실시간/준실시간 e2e 카운터**를 구현하기 위한 상세 실행계획을 정의한다.

최종 목표는 다음이다.

- 테스트셋 기준 `count accuracy = 1.00`
- 프레임 순차 처리 기반
- 후처리 없음
- 전체 영상 미래 프레임 의존 없음
- 허용 지연 `최대 1~2프레임`

---

## 전제

현재까지의 분석 결과는 [basic_jump/docs/analysis_template.md](/home/ubuntu/jump-rope-detector/basic_jump/docs/analysis_template.md)에 정리되어 있으며, 구현 계획은 다음 결론을 고정 전제로 사용한다.

- 라벨은 공중 apex가 아니라 `접지-반등 이벤트 윈도우`에 가깝다.
- `1카운트`는 `양발 접지 후 하강이 끝나고 다시 상승으로 전환되는 1회의 반등 이벤트`로 정의한다.
- 가장 유력한 전략은 `힙 주도 + 발 접지 게이트 상태기계`이다.
- 영상은 30fps CFR로 보는 것이 타당하다.
- 라벨은 파일군별 시간축 변환이 필요하다.
- 테스트셋 기준 e2e 성능을 먼저 맞춘다.

---

## 현재 코드베이스 상태

현재 저장소는 구현 코드가 거의 없고, 다음만 존재한다.

- 문서: `AGENTS.md`, `PLANS.md`, `basic_jump/docs/analysis_template.md`
- 데이터: `basic_jump/label/*.kva`, `basic_jump/video/*.mp4`
- 환경: `requirements.txt`, `scripts/setup_env.sh`

현재 없는 것:

- 라벨 파서
- 비디오 로더
- MediaPipe 추출기
- 온라인 feature 추출기
- 실시간 카운터
- 평가기
- 실험/튜닝 코드
- 테스트 코드

따라서 구현은 사실상 새 모듈 구조를 정의하는 것부터 시작해야 한다.

---

## 성공 기준

### 1차 수용 기준

- `11개 영상 전체`에 대해 최종 예측 카운트가 GT 최종 카운트와 모두 일치
- 즉 `Exact Video Count Accuracy = 11/11 = 1.00`

정의:

- `Exact Video Count Accuracy = 일치한 영상 수 / 전체 영상 수`

### 2차 진단 지표

최종 목표는 위 1차 기준이지만, 디버깅과 병목 분해를 위해 아래 지표를 함께 기록한다.

- `MAE of final count = 0`
- 이벤트 정합 F1
- 이벤트 precision / recall
- 카운트 emission delay
- video별 오차 분해

이벤트 정합은 라벨이 단일 프레임 정답이 아니라 이벤트 윈도우라는 점을 반영해, 개발 단계에서는 `±2 frame` tolerance로 진단한다. 단, 최종 합격 기준은 어디까지나 **영상별 총카운트 완전 일치**다.

---

## 비협상 제약

- 프레임을 순차 처리하며 카운트를 내야 한다.
- 미래 프레임 전체를 보는 offline peak detection은 금지한다.
- 후처리 배치 보정은 금지한다.
- 허용 가능한 지연은 `1~2프레임`이다.
- 실험/평가/파라미터 탐색은 병렬화할 수 있지만, 최종 카운터 로직은 실시간 추론 가능해야 한다.
- 실험 코드와 실제 카운터 코드는 분리한다.
- 라벨 시간축과 이상치는 평가 파이프라인에서 명시적으로 처리한다.

---

## GT(정답) 정의 정책

구현 전에 평가용 GT를 먼저 고정해야 한다. 이 단계가 흐리면 `accuracy 1.00` 목표 자체가 흔들린다.

### 라벨 시간축 정규화

- `01/02/04/08`: `Timestamp / 1000` 후 `round(sec * 30)`로 프레임 변환
- `03/05/06/07/09/10/11`: `Timestamp / 512`로 프레임 변환

### 라벨 이상치 처리 규칙

다음 규칙을 GT normalizer에 명시적으로 넣는다.

1. `Drawings`가 없는 이벤트

- 예: `01.kva` index 31
- 규칙: `카운트 이벤트는 유지`, 좌표만 missing 처리
- 이유: 시간축상 리듬이 연속적이고, count GT는 이벤트 존재가 더 중요함

2. 한 점프가 두 개의 한발 이벤트로 쪼개진 경우

- 예: `02.kva` index 5, 6 (`7854ms`, `7867ms`)
- 규칙: `1개 이벤트로 merge`
- merge 조건:
  - 연속 이벤트
  - 각 이벤트가 1개 foot point만 가짐
  - 시간 차가 `<= 1 frame` 또는 `<= 20ms`
- 이 규칙이 맞다면 `02`의 GT 카운트는 raw 16이 아니라 normalized 15가 된다.
- 현재 이 해석이 가장 타당해 보이지만, 구현 시 GT builder unit test에서 다시 고정 검증한다.

3. 한 이벤트에 4개 pencil이 중복 기입된 경우

- 예: `09.kva` index 3
- 규칙: `1개 이벤트 유지`, 좌표는 2개 대표점으로 dedup

4. 헤더 파일명 불일치

- 예: `10.kva`
- 규칙: `파일 stem 우선`, 해상도/길이/분석 결과를 보조 근거로 사용

### GT 산출물

GT builder는 최종적으로 다음을 만들어야 한다.

- video별 normalized event frame list
- video별 final count
- anomaly report
- evaluation manifest

---

## 제안 모듈 구조

실험 코드와 실제 카운터 코드를 분리하기 위해 아래 구조를 제안한다.

### 실제 구현 코드

- `src/jump_counter/__init__.py`
- `src/jump_counter/config.py`
- `src/jump_counter/types.py`
- `src/jump_counter/labels/parser.py`
- `src/jump_counter/labels/normalize.py`
- `src/jump_counter/io/video_reader.py`
- `src/jump_counter/pose/mediapipe_runner.py`
- `src/jump_counter/features/online_features.py`
- `src/jump_counter/counter/state_machine.py`
- `src/jump_counter/pipeline/e2e.py`
- `src/jump_counter/eval/metrics.py`
- `src/jump_counter/eval/evaluator.py`

### 실험/튜닝 코드

- `experiments/build_gt_manifest.py`
- `experiments/run_pose_cache.py`
- `experiments/analyze_feature_alignment.py`
- `experiments/search_counter_params.py`
- `experiments/inspect_failures.py`
- `experiments/run_regression_suite.py`

### 실행 진입점

- `scripts/run_counter.py`
- `scripts/evaluate_counter.py`
- `scripts/tune_counter.py`

### 테스트

- `tests/test_label_parser.py`
- `tests/test_label_normalizer.py`
- `tests/test_video_reader.py`
- `tests/test_online_features.py`
- `tests/test_state_machine.py`
- `tests/test_e2e_eval.py`

### 산출물 경로

- `artifacts/manifests/`
- `artifacts/pose_cache/`
- `artifacts/eval/`

`artifacts`는 재현 가능한 실험 결과를 저장하되, 최종 카운터 로직은 이 캐시에 의존하지 않도록 한다.

---

## 모듈별 역할 정의

### 1. 라벨 파서

입력:

- `basic_jump/label/*.kva`

출력:

- raw keyframe records
- raw timestamp
- raw point lists
- file-level metadata

검증:

- XML 파싱 성공
- 11개 파일 전부 로드 가능
- 구조 이상치가 보고서와 동일하게 탐지됨

### 2. 라벨 정규화기

입력:

- raw label records
- video metadata

출력:

- normalized event frame list
- normalized event count
- anomaly merge/drop logs

검증:

- `ms-like`, `tick-like` 시간축 변환이 정확함
- `02` split event merge 여부가 명시적으로 기록됨
- `10` 매칭이 `10.mp4`로 고정됨

### 3. 비디오 로더

입력:

- `basic_jump/video/*.mp4`

출력:

- sequential frame stream
- frame_idx
- timestamp_sec
- metadata

검증:

- 30fps 순차 읽기
- 랜덤 접근이 아닌 순차 인터페이스 제공
- metadata가 GT builder와 일치

### 4. MediaPipe 추출기

입력:

- frame stream

출력:

- landmark packet
- visibility
- pose detection success flag

검증:

- 전체 라벨 프레임 기준 pose detection rate 재현
- landmark missing 상황에서 graceful fallback
- 최종 카운터와 실험 캐시 모두 같은 인터페이스 사용

### 5. 온라인 feature 추출기

입력:

- landmark packet sequence

출력:

- `mean_hip_y`
- `mean_foot_y`
- `hip_vel`
- `foot_vel`
- `left_right_foot_gap`
- `left_right_foot_y_diff`
- `ground_contact_candidate`

검증:

- 모든 feature는 현재/과거 프레임만 사용
- smoothing이 causal임
- 라벨 이벤트 윈도우에서 hip/foot feature가 분석 보고서 결론을 재현

### 6. 실시간 카운터

입력:

- feature stream

출력:

- count emission event
- emitted frame index
- running total count
- optional debug state

검증:

- 상태 전이가 causal
- emission delay `<= 2 frame`
- duplicate count 억제

### 7. 평가기

입력:

- normalized GT events
- predicted emissions

출력:

- exact video count accuracy
- MAE
- event precision/recall/F1
- delay statistics
- per-video failure report

검증:

- 동일 입력에 대해 deterministic
- raw/normalized GT 차이를 별도 로그로 남김

---

## 핵심 알고리즘 설계 방향

최종 카운터는 `힙 주도 + 발 접지 게이트 상태기계`를 기본으로 한다.

### 상태기계 초안

- `INIT`
- `AIR_OR_RISING`
- `CONTACT_CANDIDATE`
- `CONTACT_CONFIRMED`
- `COUNT_EMITTED`

### 핵심 판정 흐름

1. `mean_foot_y`와 `left_right_foot_y_diff`로 양발 접지 후보를 감지
2. 접지 후보 상태에서 `mean_hip_y`의 하강 종료와 상승 전환을 탐지
3. 그 전환 시점에 `1 count` emit
4. `refractory`와 `air reset`으로 중복 카운트를 차단

### 필요한 온라인 적응

정확도 1.00을 위해 단순 고정 threshold보다 다음 적응 요소가 필요할 가능성이 높다.

- running baseline 기반 foot floor 추정
- torso length 또는 hip-ankle length 기반 y 정규화
- 최근 inter-count interval 기반 cadence-adaptive refractory
- visibility 저하 시 hip 중심 fallback

중요:

- 이 적응은 모두 `현재/과거 프레임`만 사용해야 한다.
- video 전체를 보고 threshold를 나중에 다시 맞추는 방식은 금지한다.

---

## 실행 Milestone

### M0. 구현 골격 준비

목표:

- 패키지 구조와 공통 타입, 설정 구조 정의

입력:

- 없음

출력:

- `src/`, `experiments/`, `tests/`, `scripts/` 골격

검증:

- import smoke test
- 기본 CLI 진입점 동작

### M1. GT 빌더 고정

목표:

- 평가용 정답 이벤트/정답 count를 재현 가능하게 고정

입력:

- `.kva`, `.mp4`

출력:

- normalized GT manifest

검증:

- 11개 영상 모두 manifest 생성
- anomaly 처리 로그 생성
- `02` merge 정책을 unit test로 고정

병목:

- GT 정의가 흔들리면 이후 accuracy 1.00 수치가 무의미해진다

### M2. 순차 비디오 + MediaPipe 파이프라인

목표:

- 프레임 순차 처리 기반 landmark 추출기 완성

입력:

- video path

출력:

- frame-by-frame landmark stream

검증:

- 라벨 프레임 기준 detection rate 재현
- 처리량 측정
- 캐시 없이도 e2e 실행 가능

### M3. 온라인 feature 추출기

목표:

- 상태기계가 쓸 수 있는 causal feature 정의

입력:

- landmark stream

출력:

- online feature stream

검증:

- 샘플 영상에서 라벨 윈도우와 feature phase 정렬 확인
- smoothing/velocity 계산이 causal인지 테스트

### M4. Counter v1

목표:

- 첫 번째 실시간 상태기계 구현

입력:

- feature stream

출력:

- running count
- count emission events

검증:

- `01`, `03`, `05` 같은 대표 영상에서 수동 디버깅
- emission delay 측정

### M5. 평가 리그 구축

목표:

- 전체 11개 영상에 대한 자동 평가 체인 구축

입력:

- GT manifest
- e2e counter outputs

출력:

- count accuracy
- MAE
- per-video report

검증:

- one-command evaluation 가능
- 실험 결과가 artifacts에 저장

### M6. 병렬 파라미터 탐색

목표:

- universal parameter set으로 `11/11` exact match 달성

입력:

- feature cache 또는 landmark cache
- search space

출력:

- best parameter set
- leaderboard

검증:

- 16 CPU 병렬 탐색
- best config가 실제 e2e sequential run에서도 동일 결과 재현

튜닝 대상:

- smoothing 강도
- contact threshold
- hip velocity hysteresis
- min contact frames
- min airborne frames
- refractory frames
- visibility fallback 규칙
- cadence adaptation 강도

### M7. 실패 케이스 분해

목표:

- 정확도 1.00이 안 나올 경우, 병목을 구조적으로 분해

입력:

- 실패 영상 리스트

출력:

- failure taxonomy

검증:

- 각 실패를 아래 범주 중 하나로 분류

분해 범주:

- GT 문제
- timestamp normalization 문제
- pose missing 문제
- feature phase drift 문제
- fast cadence duplicate 문제
- low visibility fallback 문제
- state hysteresis 문제

### M8. 리그레션 고정

목표:

- accuracy 1.00 달성 후 회귀 방지

입력:

- best config

출력:

- golden expected counts
- regression tests

검증:

- 코드 변경 후 11개 전부 재검증
- `Exact Video Count Accuracy = 1.00` 유지

---

## 정확도 1.00 달성을 위한 튜닝 전략

### 원칙

- 최종 로직은 causal
- 튜닝은 병렬/offline 가능
- 탐색 대상은 파라미터이지, 미래 프레임 의존 알고리즘이 아니다

### 단계별 전략

1. GT부터 고정

- 라벨 정규화와 anomaly merge 규칙을 먼저 고정

2. landmark cache 생성

- MediaPipe 추론 비용을 분리해 탐색 속도를 높임
- 단, 최종 평가는 반드시 raw video → sequential landmark → counter의 e2e로 다시 수행

3. universal threshold 먼저 탐색

- video별 임시 threshold는 사용하지 않음
- 먼저 전역 파라미터로 `11/11`이 가능한지 확인

4. 안 되면 causal adaptation 추가

- running floor estimate
- normalized hip amplitude
- cadence-adaptive refractory
- visibility-aware fallback

5. 다시 global search

- 적응 규칙을 포함한 전역 파라미터 탐색 수행

6. 마지막에 failure-specific debugging

- `09`, `10` 같은 빠른 리듬/세로 영상이 우선 점검 대상이 될 가능성이 높음

### 병렬화 포인트

- video별 pose cache 생성 병렬화
- 파라미터 조합 평가 병렬화
- regression suite 병렬화

### 탐색 공간 축소 원칙

- smoothing 길이는 `1~3 frame` 수준만 탐색
- refractory는 jump cadence 기반 합리적 범위만 탐색
- contact threshold는 normalized coordinate 기준 좁은 범위 탐색

---

## 정확도 1.00이 안 나올 때의 우선 병목

1. GT normalizer가 잘못되어 정답 count 자체가 흔들리는 경우
2. `02` split event 같은 anomaly가 평가를 오염시키는 경우
3. 빠른 cadence에서 한 점프를 2번 세는 duplicate 문제
4. 접지 구간이 길어져 한 점프를 0번 세는 miss 문제
5. `10.mp4`류의 visibility 저하로 foot gate가 흔들리는 경우
6. ms-like 그룹에서 frame rounding으로 emission timing이 밀리는 경우

우선순위는 항상 다음 순서로 둔다.

- GT 문제
- timestamp 문제
- state machine 문제
- MediaPipe 문제

---

## 과적합 위험

테스트셋 기준 1.00을 먼저 만드는 것은 현재 목표에 부합하지만, 동시에 과적합 위험을 명확히 안고 간다.

- 11개 영상 전체를 보고 threshold를 맞추면 데이터셋 특화 규칙이 생길 수 있다.
- 카메라 구도, 피사체 키, 신발 높이, 배경, 점프 리듬 분포가 제한적이다.
- `02` anomaly merge 같은 규칙이 현재 셋에만 맞을 수 있다.

따라서 구현 단계에서는 아래를 반드시 분리한다.

- `테스트셋 최적화용 파라미터 탐색`
- `일반화 가능한 causal 규칙`

---

## 일반화 리스크

- 더 낮은 해상도
- 발 일부 잘림
- 비정면 촬영
- 카메라 흔들림
- 더 느리거나 더 빠른 cadence
- 초보자의 비대칭 jump
- rope가 보이지 않거나 상체 흔들림이 큰 경우

현재 계획은 이 리스크를 완전히 해소하지 않는다. 현재 단계의 목표는 **테스트셋 e2e 정확도 1.00을 재현 가능한 방식으로 달성하는 것**이다.

---

## 이번 단계의 산출물

이번 계획 단계가 끝나면 다음이 명확해야 한다.

- 필요한 모듈 구조
- GT builder 정책
- milestone별 입력/출력/검증 방식
- 평가 기준
- 튜닝 전략
- 병목 분해 순서
- PLANS 문서 갱신

---

## 현재 상태

- [x] 분석 결과 문서화
- [x] 구현 목표 재정의
- [x] 모듈 구조 초안 정의
- [x] milestone 정의
- [x] GT 정책 정의
- [x] 튜닝 전략 정의
- [x] 리스크 분리
- [x] 실제 구현 시작
- [x] MVP label parser / video loader / MediaPipe / realtime counter 구현
- [x] label-start window evaluator 반영
- [x] `/basic_jump/video` 기준 exact count accuracy 1.00 달성
- [x] 실시간 스트림 검증용 runner 추가

## 다음 액션

- 현재 MVP 로직을 유지한 채 predicted event frame과 GT event frame의 정렬 오차를 계량화한다.
- `basic_jump/` 코드를 `src` 기반 구조로 옮길지 여부는 정확도 회귀 테스트를 먼저 만든 뒤 결정한다.
- 카메라 실측에서 `full landmark ready 1초 -> 3초 countdown` UX가 과도하게 엄격한지 확인한다.
