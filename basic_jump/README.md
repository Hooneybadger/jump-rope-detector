# basic_jump Counter

## 동작 원리

### 1. 무엇을 1카운트로 보는가

`양발이 접지된 상태에서 몸이 내려갔다가 다시 올라가기 시작하는 반등 전환점`을 1카운트로 본다.

### 2. 어떤 신호를 보는가

- `mean_hip_y`: 좌우 hip의 평균 y. 몸 중심의 상하 움직임을 본다.
- `mean_foot_y`: ankle, heel, foot index를 합친 발 y. 양발이 바닥 근처인지 본다.
- `leg_length`: hip-ankle 길이. 사람 크기와 카메라 구도 차이를 정규화하는 기준이다.
- `slow baseline`: hip와 foot의 느린 기준선이다. 사람 전체가 카메라 쪽으로 오거나 멀어질 때 생기는 저주파 드리프트를 분리하는 데 쓴다.

핵심 구조는 `힙 주도 + 발 접지 게이트`다.  
힙은 반등 시점을 잡고, 발은 지금이 실제 점프 접지 구간인지 확인한다.

양발이 바닥 근처에 있는지는 절대 픽셀값으로 보지 않는다.

- `바닥 밴드`는 "지금 이 영상에서 발이 바닥에 닿아 있을 때 나타나는 y 높이 영역"을 뜻한다.
- 카메라 높이, 사람 키, 프레이밍이 다르기 때문에 바닥을 고정된 한 좌표로 두지 않고, 최근 발 위치로 계속 갱신한다.
- 최근 발 y를 바탕으로 현재의 `바닥 밴드`를 온라인으로 추정한다.
- 이때 원래 y를 그대로 쓰지 않고, `slow baseline`을 뺀 발 residual 기준으로 바닥 밴드를 추적한다.
- 현재 `mean_foot_y`가 그 바닥 밴드 근처에 있으면 `접지 후보`로 본다.
- 동시에 좌우 발 높이 차가 작아야 한다.

즉, `발 평균 높이`와 `좌우 대칭성`을 함께 봐서 양발 접지 상태를 판단한다.

### 3. 카운트는 어떤 순서로 올라가는가

1. 양발이 바닥 근처에 있고 좌우 발 높이 차가 작으면 `접지 상태`로 본다.
2. 접지 상태에서 힙 residual이 먼저 아래로 내려가면 `반등 준비`로 본다.
3. 그 다음 힙 움직임이 `하강 -> 상승`으로 바뀌는 첫 프레임에서 카운트를 1 올린다.
4. 직후에는 `lock` 상태로 들어가 같은 점프를 두 번 세지 않게 막는다.

즉, `양발 접지 -> 하강 -> 상승 전환 -> count` 순서다.

### 4. 왜 손동작이나 발장난을 덜 세는가

- 손만 흔든 경우: 발 움직임 범위가 부족하면 reject한다.
- 발만 까딱한 경우: 힙 움직임 범위가 부족하면 reject한다.
- 발 움직임이 힙 움직임보다 과도하게 큰 경우: `foot_to_hip_ratio`와 `foot_dominant_low_hip` guard로 reject한다.
- 같은 점프를 두 번 세는 경우: `lock + min gap`으로 차단한다.
- 빠른 리듬에서는: 최근 jump 간격을 보고 `min gap`을 자동으로 줄여 undercount를 막는다.

여기서 두 용어의 역할은 다르다.

- `lock`: 한 번 count한 직후 바로 다음 후보를 받지 않고, 몸이 다시 충분히 올라갔다가 다음 하강 cycle이 시작된 것이 확인될 때까지 기다리는 상태다.
- `min gap`: 연속 두 count 사이에 필요한 최소 프레임 간격이다. unified 엔진의 최종 accept 단계에서 마지막 accepted count와 현재 후보 사이 간격을 보고, 너무 가까우면 count를 올리지 않는다.

즉, `lock`은 상태기계 내부 중복 방지이고, `min gap`은 최종 accepted count 단계의 안전장치다.

같은 점프를 연속으로 세지 않게 막는 방식은 아래와 같다.

1. 한 번 count가 올라가면 상태기계는 바로 `lock` 상태로 들어간다.
2. 이 상태에서는 같은 접지 구간 안에서 생기는 작은 흔들림이나 추가 velocity 반전을 무시한다.
3. 몸이 다시 올라가고, 다음 jump cycle로 넘어가는 하강 흐름이 확인된 뒤에만 다음 count를 받을 수 있게 푼다.

`min gap`은 그 위에 한 번 더 거는 안전장치다.

- 마지막으로 받아들인 count 이후 프레임 수를 센다.
- 그 간격이 최소 기준보다 짧으면, 상태기계가 count 후보를 냈더라도 최종 count는 올리지 않는다.
- 기본값은 보수적으로 잡고, 빠른 리듬이 감지되면 자동으로 줄인다.

이걸 강제했을 때의 trade-off도 있다.

- `lock`이 너무 길면: 실제 다음 점프가 시작됐는데도 이전 점프로 보고 놓칠 수 있다.
- `min gap`이 너무 크면: cadence가 빠른 사람의 진짜 jump까지 중복으로 오해하고 버릴 수 있다.
- 반대로 너무 짧으면: 한 점프 안의 흔들림을 두 번 세는 double-count가 다시 늘어난다.

현재 구현은 이 균형을 맞추기 위해, 평소에는 보수적으로 막고 빠른 cadence가 확인되면 `min gap`을 자동 완화하는 쪽을 사용한다.

추가로 현재 엔진은 아래 두 guard를 함께 쓴다.

- `foot_to_hip_ratio`: 발 range가 힙 range를 비정상적으로 크게 압도하면 reject한다.
- `foot_dominant_low_hip`: 발 range는 큰데 힙 range와 최근 힙 range가 약하면, 발장난이나 꼬리 흔들림으로 보고 reject한다.


### 5. 언제 카운팅을 시작하는가

- jump에 필요한 핵심 landmark(hip, knee, ankle, heel, foot index)가 약 `1초` 안정적으로 보이면 준비 완료로 본다.
- 그 뒤 `3초` countdown 후 카운팅을 시작한다.
- ready hold와 countdown 동안에는 실제 count는 올리지 않고, 발 바닥 밴드와 motion history, 초기 jump 상태를 미리 적응시킨다.
- 그래서 시작 자세와 실제 점프 자세의 발 위치가 조금 달라도, 카운트 시작 시점에 기준선이 더 빨리 맞춰진다.
- 준비나 countdown 중 landmark가 잠깐 흔들리는 정도는 흡수한다.
- landmark가 더 길게 끊기면 다시 준비 상태로 돌아간다.

## Run

카메라 입력:

```bash
bash scripts/setup_env.sh
source activate
python basic_jump/run_realtime_counter.py --source 0
```

시연 영상 저장:

```bash
source activate
python basic_jump/run_realtime_counter.py --source 0 --save-output basic_jump/artifacts/realtime_demo.mp4
```
