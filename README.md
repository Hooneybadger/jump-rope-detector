# jump-rope-detector

MediaPipe Pose 기반 실시간 줄넘기 카운터.  
모아뛰기 · 번갈아뛰기 · 이중뛰기 세 종목을 지원한다.

---

## 환경 설정

```bash
bash scripts/setup_env.sh
source activate
```

---

## 실행

### 웹 앱 (streamlit)

브라우저에서 카메라를 열어 실시간 카운팅을 진행한다.

```bash
streamlit run app.py
```

1. 브라우저에서 카메라 접근 허용
2. 종목 카드 클릭 → 카운팅 시작
3. **Stop / 결과 보기** 버튼으로 종료

> 원격 서버에서 실행할 경우 HTTPS 또는 `localhost` 환경이 필요하다 (WebRTC 보안 정책).

### 데스크톱 앱

로컬 환경에서 OpenCV 창으로 실행한다.

```bash
python run.py
# 영상 파일 입력 시
python run.py --source /path/to/video.mp4
```

1. `1` / `2` / `3` 키 또는 마우스 클릭으로 종목 선택
2. 카운팅 진행
3. `q` 키로 종료 → 결과 확인 후 메뉴 복귀

> GUI 디스플레이가 없는 서버 환경에서는 실행되지 않는다.

---

## 종목

| 종목 | 설명 | 상세 문서 |
|------|------|-----------|
| 모아뛰기 | 양발 모아 동시에 뛰기 | [basic_jump/README.md](basic_jump/README.md) |
| 번갈아뛰기 | 왼발·오른발 교대로 뛰기 | [alternating_jump/README.md](alternating_jump/README.md) |
| 이중뛰기 | 한 번 점프에 줄 두 번 통과 | [double_jump/README.md](double_jump/README.md) |

### 종목별 단독 실행

```bash
python basic_jump/run_realtime_counter.py --source 0
python alternating_jump/run_realtime_counter.py --source 0
python double_jump/run_realtime_counter.py --source 0
```
