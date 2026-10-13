# GPU 전환 검증 결과

현재 앱은 브라우저 GPU 추론·관절 전송을 목표 60fps로 실행하고, 기존 카운팅 엔진 입력은 20Hz로 선택합니다. 이중뛰기의 30Hz 보간은 유지합니다.

- 최종 정확도·처리 시간·FPS: [60fps/README.md](60fps/README.md)
- 영상별 결과: [60fps/comparison.csv](60fps/comparison.csv)
- 종목별 기계 판독용 집계: [60fps/summary.json](60fps/summary.json)
- 프레임별 실측 시간과 계수 이벤트: [60fps/comparison.json](60fps/comparison.json)
- 실제 앱의 FPS·카메라 도착률: [60fps/live-fps.json](60fps/live-fps.json)
- 최초 20fps GPU/CPU 비교: [20fps/README.md](20fps/README.md)
- 엔진 보존 검사: [engine-preservation.json](engine-preservation.json)
- 모델·WASM 자산 검사: [asset-verification.json](asset-verification.json)
- 검증·정리 기록: [validation.json](validation.json)

카운팅 상태기계·임계값·분류기 가중치는 변경하지 않았습니다. 모아뛰기 일치도는 하락했으며 번갈아뛰기·이중뛰기는 개선됐습니다. Headless Chrome의 60fps 영상 재생에서 실제 분석률은 48.5~51.0fps였고 카메라 콜백 도착률도 비슷했습니다. 물리 웹캠 또는 원격 네트워크에서의 60fps 달성을 검증한 결과는 아닙니다.

원본 영상·라벨은 cache/basic, cache/alternating, cache/double에 보존했습니다. 이 폴더는 Git 및 Docker 빌드에서 제외됩니다. 임시 Python/Node 환경·가짜 웹캠 Y4M·중복 JSON·이전 스크린샷은 검증 후 삭제했습니다.

재실행은 최종 보고서의 절차를 따르세요. 브라우저 FPS 검증 도구는 tools/check_live_fps.cjs이며, tools/make_benchmark_camera.py로 60fps 원본의 Y4M 재생 입력을 생성할 수 있습니다. FPS_ORIGIN, FPS_USERNAME, FPS_PASSWORD, FPS_CAMERA_Y4M 환경 변수로 로컬 테스트 서버·테스트 계정·영상 경로를 지정합니다. 실제 운영 계정 대신 별도의 테스트 DB와 계정을 사용합니다.
