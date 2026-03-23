# basic_jump Counter

## Dataset Eval

```bash
source activate
PYTHONPATH=. python3 basic_jump/run_dataset_eval.py
PYTHONPATH=. python3 basic_jump/run_dataset_eval.py --grid-search
```

기본 평가는 `first_label - 15 frame`부터 `last_label`까지를 카운트 윈도우로 사용하고, 그 직전 `4 frame`은 warmup으로만 처리한다.

결과 요약은 `basic_jump/artifacts/dataset_eval_results.json`에 저장된다.

기존 호환 명령도 유지된다.

```bash
PYTHONPATH=. python3 basic_jump/run_mvp.py
```

## Realtime

카메라 입력:

```bash
source activate
PYTHONPATH=. python3 basic_jump/run_realtime_counter.py --source 0
```

파일 스트림으로 순차 검증:

```bash
source activate
PYTHONPATH=. python3 basic_jump/run_realtime_counter.py --source basic_jump/video/09.mp4
```

시연 영상 저장:

```bash
source activate
python basic_jump/run_realtime_counter.py --source 0 --save-output basic_jump/artifacts/realtime_demo.mp4
```

시작 조건:

- 전체 Pose landmark가 `1초` 연속으로 보이면 준비 완료
- 그 뒤 `3초` countdown 후 카운트 시작
- 준비/카운트다운 중 짧은 landmark flicker는 흡수하고, 연속 dropout일 때만 준비 단계로 돌아감
- realtime에서는 추가로 `hip/foot motion validator`와 `min gap`을 적용해 손-only, 발-only, double-count를 줄인다
- 빠른 cadence가 감지되면 realtime 필터가 `raw candidate interval`을 기준으로 `min gap`과 `recent hip` 임계치를 자동 완화한다
- 고정 임계치로 비교하려면 `--disable-adaptive-filter`를 사용한다
