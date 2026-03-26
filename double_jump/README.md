# double_jump Counter

`double_jump` 카운터는 이제 점프 높이보다 손목 회전 위상을 우선해서 2단 뛰기를 판정한다.

## 핵심 아이디어

- `foot` 신호는 이륙과 착지 구간을 자르는 용도로만 쓴다.
- `wrist`와 `elbow`, `shoulder`로 몸 기준 좌표계를 만든다.
- 각 손목의 회전 위상을 추적하고, 공중 구간에서 누적 회전량이 약 2회전인지 본다.
- 좌우 손목의 회전량 균형과 위상 동기성, 회전 cadence, FFT 집중도를 함께 검증한다.

즉, 성공 기준은 더 이상 "얼마나 높이 떴는가"가 아니라 "공중 구간 안에서 손목이 실제로 두 번 회전에 가까운 패턴을 만들었는가"다.

## 현재 파이프라인

1. MediaPipe Pose에서 shoulder, elbow, wrist, hip, foot landmark를 읽는다.
2. `mean_foot_y`를 low-pass 한 뒤 online floor와 contact gate를 만든다.
3. contact gate를 벗어난 프레임이 연속으로 나오면 airborne segment를 연다.
4. 손목 좌표를 어깨 중심 기준으로 평행이동하고, 어깨선 방향으로 회전해 body-centered 좌표로 바꾼다.
5. 각 손목의 `elbow -> wrist` 벡터 각도를 계산하고, 프레임 간 위상 변화량을 누적한다.
6. 착지 시점에 아래 특징량으로 최종 판정한다.

## 최종 판정 특징량

- `left_rotation_count`
- `right_rotation_count`
- `wrist_rotation_count`
  좌우 누적 회전량 평균
- `wrist_rotation_balance`
  좌우 회전량 균형
- `phase_sync_ratio`
  좌우 위상 동기성
- `wrist_rotation_cadence_hz`
  공중 구간 내 평균 회전 cadence
- `wrist_fft_peak_hz`
- `wrist_fft_power_ratio`

## 왜 이 방식이 더 맞는가

- 속도 peak 개수보다 위상 누적량이 실제 회전 횟수에 더 직접적이다.
- 점프 높이, hip lift, wrist-to-jump ratio를 제거해서 높이 편차의 영향을 줄였다.
- 양손 줄넘기 특성상 좌우 손목이 같이 돌아야 하므로, phase sync와 rotation balance가 좋은 억제 신호가 된다.
- FFT는 주신호가 아니라 회전 리듬 품질 검증용 보조 신호로 남겼다.

## 주요 설정값

- `min_airborne_frames`
- `max_airborne_frames`
- `takeoff_confirm_frames`
- `min_wrist_rotation_count`
- `max_wrist_rotation_count`
- `min_wrist_rotation_balance`
- `min_phase_sync_ratio`
- `min_rotation_cadence_hz`
- `min_fft_peak_hz`
- `min_fft_power_ratio`

## 참고

실시간 디버그 출력은 기존 `jump_height`, `hip_lift`, `wrist_peak` 대신 회전량, 동기성, cadence를 표시한다.
