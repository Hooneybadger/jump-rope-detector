# PLANS.md

## 목표

`/basic_jump/video` 테스트셋 평가와 realtime 실행이 **같은 카운트 로직**을 사용하도록 통합한다.

최종 목표는 다음 두 가지를 동시에 만족하는 것이다.

- 테스트셋 11개 영상 기준 `Overall Count Accuracy = 1.00`
- realtime에서도 같은 엔진이 causal하게 동작

보조 지표:

- 테스트셋 11개 영상 기준 `Exact Video Count Accuracy = 1.00`

핵심 원칙:

- `평가용 엔진`과 `실사용 엔진`의 count emission 규칙은 하나여야 한다.
- 같은 입력 signal이 들어오면 dataset eval과 realtime이 같은 frame에서 같은 count를 내야 한다.
- `start gate`와 `UI`는 세션 제어/입출력 계층으로 분리할 수 있지만, count를 올리는 판단식은 분리하지 않는다.

---

## 현재 진단

현재 구현은 완전히 분리된 두 엔진은 아니지만, **count emission 경로가 하나로 통합되어 있다고 보기도 어렵다.**

현재 구조:

- 공통:
  - `MediaPipe -> SignalFrame`
  - `RealtimeCounterEngine`
- dataset eval:
  - `run_counter_on_signals()`에서 `RealtimeCounterEngine`만 사용
- realtime:
  - `RealtimeCounterEngine`
  - 그 뒤에 `RealtimeCountFilter`
  - 그 바깥에 `RealtimeStartGate`

즉 현재는:

- dataset 평가는 `state machine raw event`를 센다.
- realtime은 `state machine raw event`를 다시 `RealtimeCountFilter`로 거른 뒤 센다.

이 구조의 문제:

- dataset `1.00`은 raw event 기준 성능이고
- realtime 카운트는 filtered event 기준 성능이라
- 둘이 같은 지표라고 보기 어렵다.

사용자 지적은 타당하다.

> 상태엔진과 realtime 카운트 로직이 다르면, 상태엔진 평가지표를 realtime 평가지표로 사용할 수 없다.

따라서 앞으로의 목표는:

- `state machine + filter`를 합쳐 **하나의 단일 카운터 엔진**으로 만들고
- dataset eval도 그 엔진을 그대로 쓰게 만드는 것이다.

---

## 통합 원칙

### 1. 단일 count emission path

카운트는 오직 한 곳에서만 올라가야 한다.

- 현재의 `RealtimeCounterEngine.step()`와
- 현재의 `RealtimeCountFilter.accept()`

이 둘을 분리해서 유지하지 않고, 하나의 `step()` 안에서 최종 accepted count가 결정되게 만든다.

### 2. 단일 파라미터 집합

평가와 realtime이 같은 로직을 쓴다면, 임계치도 한 묶음이어야 한다.

따라서 아래 파라미터는 하나의 config로 합친다.

- 상태기계 파라미터
  - EMA
  - contact margin
  - symmetry
  - velocity threshold
  - refractory
- realtime 보호 파라미터
  - hip/foot range 조건
  - recent hip 조건
  - min gap
  - adaptive cadence 조건

### 3. start gate는 count logic가 아니다

`RealtimeStartGate`는 count emission 규칙이 아니라 세션 시작 조건이다.

따라서:

- realtime에서는 유지 가능
- dataset eval에서는 label window가 시작 구간을 대신하므로 적용하지 않아도 됨

단, start gate 때문에 count 판정식 자체가 달라져서는 안 된다.

### 4. debug 정보와 count 판정 분리

통합 이후에도 디버깅은 필요하다.

하지만 다음은 분리한다.

- `count를 올릴지 말지`
- `왜 reject됐는지 설명하는 debug metadata`

즉, 최종 엔진은 하나이고, debug 정보는 부가 출력으로만 남긴다.

---

## 목표 구조

통합 후 목표 구조는 아래와 같다.

### 단일 엔진

- 파일:
  - `basic_jump/counter_engine.py`
- 새 역할:
  - `UnifiedCounterConfig`
  - `UnifiedCounterEngine`
  - `step(signal) -> AcceptedEvent | None`
  - optional debug snapshot

이 엔진 내부에서 모두 처리한다.

- contact state machine
- rebound lock
- motion window feature
- min gap
- adaptive cadence
- final accept / reject

### dataset eval

- `run_counter_on_signals()`가 `UnifiedCounterEngine`을 직접 사용
- 더 이상 dataset 경로에서 raw state-machine event만 세지 않음
- 평가 결과는 realtime과 같은 acceptance logic 기준이 됨

### realtime

- `run_realtime_counter.py`는 아래만 담당
  - camera/file input
  - MediaPipe extraction
  - start gate
  - UI
  - save-output
- count logic은 unified engine만 호출

즉 realtime runner에서는 `RealtimeCountFilter`가 사라지거나, 최소한 별도 count 판단기로 존재하지 않게 된다.

---

## 마일스톤

### M1. 기준선 고정

목표:

- 현재 동작을 통합 전 baseline으로 고정

기록할 기준:

- dataset eval exact accuracy
- `02_realtime` 결과
- `01_realtime` 결과
- 주요 reject reason

입력:

- `basic_jump/video/*.mp4`
- `basic_jump/label/*.kva`
- `01_realtime.mp4`, `02_realtime.mp4`

출력:

- baseline 수치 메모
- 통합 전/후 비교 기준

검증:

- baseline 로그가 재현 가능해야 함

### M2. 엔진 통합

목표:

- `RealtimeCountFilter` 로직을 `counter_engine.py` 안으로 합친다.

해야 할 일:

- `EngineConfig`와 `RealtimeFilterConfig`를 하나로 통합
- state machine raw candidate와 final accepted count를 같은 엔진에서 처리
- `run_counter_on_signals()`가 통합 엔진을 사용하게 변경
- realtime runner에서 별도 accept 단계 제거

입력:

- `SignalFrame`

출력:

- accepted `CounterEvent`
- optional debug decision

검증:

- 같은 입력 신호에 대해 dataset eval과 realtime이 같은 count frame을 내는지 확인

### M3. 재튜닝

목표:

- 통합 엔진 기준으로 다시 `1.00`을 맞춘다.

해야 할 일:

- 현재 dataset `1.00`이 깨지는지 확인
- 통합 config 기준으로 small sweep
- fast cadence와 false positive 억제의 균형 재조정

검증:

- `Overall Count Accuracy = 1.00`
- `Exact Video Count Accuracy = 1.00`
- `Total Abs Error = 0`

### M4. realtime 회귀 검증

목표:

- 통합 엔진이 realtime에서도 망가지지 않았는지 확인

검증 대상:

- `02_realtime`: 빠른 cadence undercount 재발 여부
- `01_realtime`: false positive 폭증 여부
- counting start 이후 frame-level count emission 안정성

성공 기준:

- `02_realtime`에서 빠른 연속 점프를 놓치지 않음
- `01_realtime`에서 false positive가 크게 증가하지 않음

### M5. 문서/실행 경로 정리

목표:

- README, 기술 문서, 실행 스크립트가 새 구조를 정확히 설명하도록 정리

수정 대상:

- `basic_jump/README.md`
- `basic_jump/docs/analysis_template.md`
- 필요 시 `run_dataset_eval.py`, `run_realtime_counter.py`

---

## 검증 기준

통합 이후 검증은 아래 순서로 본다.

### 1. 정합성

- dataset eval과 realtime이 **같은 count logic**을 호출하는가

### 2. 정답 성능

- 11개 테스트셋 overall count accuracy = `1.00`
- 11개 테스트셋 total abs error = `0`
- 11개 테스트셋 exact accuracy = `1.00`

### 3. 인과성

- centered window 없음
- 미래 프레임 전체 의존 없음
- frame 순차 처리만으로 동작

### 4. realtime UX

- counting start 이후 count emission이 지연 없이 자연스러운가
- 빠른 cadence에서 undercount / double-count가 과도하지 않은가

---

## 리스크

### 1. dataset 1.00 붕괴 위험

현재 dataset `1.00`은 unified engine과 label window 조합 기준으로 맞춰져 있다.

통합 필터가 들어오면:

- dataset 카운트가 줄어들거나
- 일부 영상에서 miss가 생길 수 있다.

이건 통합 과정에서 가장 먼저 확인해야 하는 리스크다.

### 2. false positive 억제 약화 위험

반대로 dataset exact를 지키려고 filter를 약하게 만들면 realtime false positive가 다시 늘어날 수 있다.

### 3. start gate와 count logic의 경계 혼동

start gate는 세션 제어일 뿐 count logic이 아니다.

이 둘을 다시 섞어버리면:

- 평가와 실사용 비교가 다시 흐려진다.

따라서 통합 범위는 `count emission logic`으로 한정해야 한다.

---

## 현재 상태

- 통합 필요성: 확인 완료
- 구조 차이 분석: 완료
- 계획 문서 갱신: 완료
- 코드 통합: 완료
- 현재 unified engine 기본값 결과:
  - dataset eval `Overall Count Accuracy = 1.0000`
  - dataset eval `Exact Video Count Accuracy = 1.0000`
  - dataset eval `Total Abs Error = 0`
  - dataset eval `Total Count = 1394 / 1394`
  - `01_realtime = 3`
  - `02_realtime = 20`
- 현재 남은 핵심 리스크:
  - 테스트셋 밖 cadence/구도 변화에서 현재 threshold가 그대로 유지되는지 검증이 부족함
  - `RealtimeStartGate`는 count logic와 분리되어 있지만, 실제 카메라 환경에서 landmark flicker가 심하면 start UX가 달라질 수 있음

---

## 다음 액션

다음 구현 턴의 우선순위는 아래 순서다.

1. 현재 `Total Abs Error = 0` 상태를 regression baseline으로 고정
2. 실제 카메라 입력에서 start gate와 count emission이 계속 안정적인지 smoke test 확대
3. `01_realtime`, `02_realtime` 외 추가 positive/negative realtime 샘플로 회귀 세트 확장
4. threshold 민감도 범위를 기록해서 향후 회귀 시 어떤 파라미터가 깨졌는지 빠르게 찾을 수 있게 정리
