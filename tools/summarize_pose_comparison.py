"""Summarize raw browser/CPU comparison data without cancelling opposite errors."""
import csv
import json
import statistics
import sys
from pathlib import Path

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "benchmark_results" / "60fps"
NAMES = {"basic": "모아뛰기", "alternating": "번갈아뛰기", "double": "이중뛰기"}


def stats(values):
    ordered = sorted(values)
    def percentile(p):
        index = (len(ordered)-1)*p
        low = int(index)
        high = min(low+1, len(ordered)-1)
        return ordered[low]+(ordered[high]-ordered[low])*(index-low)
    return dict(mean=statistics.mean(values), p50=percentile(0.5), p95=percentile(0.95))


def main():
    data = json.loads((OUT / "comparison.json").read_text(encoding="utf-8"))
    rows, summaries = [], []
    for r in data["results"]:
        cpu, gpu = r["legacy"]["timings"][10:], r["timings"][10:]
        rows.append(dict(id=r["id"], mode=r["mode"], gt=r["gt"], cpu=r["legacy"]["count"], gpu=r["gpu"]["count"],
                         cpu_error=r["legacy"]["error"], gpu_error=r["gpu"]["error"],
                         cpu_mean_ms=stats([t["total_ms"] for t in cpu])["mean"],
                         gpu_mean_ms=stats([t["compute_ms"] for t in gpu])["mean"],
                         gpu_inference_ms=stats([t["inference_ms"] for t in gpu])["mean"],
                         gpu_server_ms=stats([t["server_ms"] for t in gpu])["mean"],
                         init_ms=r["initMs"], first_inference_ms=r["timings"][0]["inference_ms"]))
    for mode in NAMES:
        records = [r for r in data["results"] if r["mode"] == mode]
        selected = [r for r in rows if r["mode"] == mode]
        cpu = [t for r in records for t in r["legacy"]["timings"][10:]]
        gpu = [t for r in records for t in r["timings"][10:]]
        gt = sum(r["gt"] for r in selected)
        cpu_abs = sum(abs(r["cpu_error"]) for r in selected)
        gpu_abs = sum(abs(r["gpu_error"]) for r in selected)
        cpu_ms, gpu_ms = stats([t["total_ms"] for t in cpu]), stats([t["compute_ms"] for t in gpu])
        summaries.append(dict(mode=mode, clips=len(records), gt=gt,
                              cpu_count=sum(r["cpu"] for r in selected), gpu_count=sum(r["gpu"] for r in selected),
                              cpu_abs_error=cpu_abs, gpu_abs_error=gpu_abs,
                              cpu_mae=cpu_abs/len(records), gpu_mae=gpu_abs/len(records),
                              cpu_agreement_pct=100*(1-cpu_abs/gt), gpu_agreement_pct=100*(1-gpu_abs/gt),
                              cpu_ms=cpu_ms, gpu_ms=gpu_ms,
                              gpu_inference_ms=stats([t["inference_ms"] for t in gpu]),
                              gpu_server_ms=stats([t["server_ms"] for t in gpu]),
                              gpu_local_http_roundtrip_ms=stats([t["roundtrip_ms"] for t in gpu]),
                              payload_bytes=stats([t["payload_bytes"] for t in gpu]),
                              speedup=cpu_ms["mean"]/gpu_ms["mean"], timed_frames=len(gpu)))
    with (OUT / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(browser=data["browser"], date=data["date"], renderer=data["results"][0]["info"]["renderer"],
                  summaries=summaries, rows=rows)
    (OUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    report = ["# 브라우저 GPU 전환 비교", "", f"실행: {data['date']}, Chrome {data['browser']}",
              "", f"GPU: {result['renderer']}", "", "CPU: AMD Ryzen 5 7500F, 12 logical processors.", "",
              "## 종목별 결과", "",
              "| 종목 | 영상 | 정답 | CPU 계수 | GPU 계수 | CPU 횟수 일치도 | GPU 횟수 일치도 | CPU 평균 처리 | GPU 평균 처리 | 속도 비율 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        report.append(f"| {NAMES[s['mode']]} | {s['clips']} | {s['gt']} | {s['cpu_count']} | {s['gpu_count']} | {s['cpu_agreement_pct']:.2f}% | {s['gpu_agreement_pct']:.2f}% | {s['cpu_ms']['mean']:.2f}ms | {s['gpu_ms']['mean']:.2f}ms | {s['speedup']:.2f}배 |")
    report += ["", "횟수 일치도 = 100 × (1 − 영상별 절대 횟수 오차 합 / 정답 횟수 합). 과소·과대 계수를 상쇄하지 않는다. 이벤트별 정확도나 종목 분류 정확도가 아니다.",
               "", "## 시간의 측정 범위", "",
               "| 종목 | CPU p50 / p95 | GPU p50 / p95 | GPU 모델 평균 | 서버 평균 |",
               "|---|---:|---:|---:|---:|"]
    for s in summaries:
        report.append(f"| {NAMES[s['mode']]} | {s['cpu_ms']['p50']:.2f} / {s['cpu_ms']['p95']:.2f}ms | {s['gpu_ms']['p50']:.2f} / {s['gpu_ms']['p95']:.2f}ms | {s['gpu_inference_ms']['mean']:.2f}ms | {s['gpu_server_ms']['mean']:.3f}ms |")
    report += ["", "CPU 시간은 JPEG 디코딩 + 기존 Pose 추론 + 엔진이다. GPU 시간은 실제 BrowserPose/Worker의 ImageBitmap 생성·전달·추론 + 서버 좌표 검증·신호 변환·엔진이다. 영상 읽기·브라우저 PNG 로드·네트워크는 모두 제외한다. GPU의 모델 시간·서버 시간·로컬 HTTP 왕복도 따로 저장했다.",
               "", "GPU 첫 10개 추론과 CPU 첫 10개 프레임은 시간 집계에서만 제외한다. 모델 생성 시간(init_ms)과 첫 프레임 추론(first_inference_ms)은 CSV에 별도 기록한다. 최초 셰이더 준비 때문에 첫 프레임은 정상 상태보다 느릴 수 있다.",
               "", "실제 NVIDIA/Direct3D11 하드웨어 렌더러를 확인한 Headless Chrome 실행이다. 이 PC의 GPU를 다른 Python 작업도 사용 중이었고 해당 프로세스는 중지하지 않았다. 부하가 변하므로 전용 GPU 환경·모바일의 속도로 일반화하지 않는다.",
               "", "## 정확도 조건과 한계", "",
               "- 모아뛰기 10쌍: jummpingRope.zip/.z01의 fsm/basic_jump/input/video 및 input/label을 cache/basic에 추출. 번갈아뛰기 6쌍·이중뛰기 4쌍: cache/alternating 및 cache/double의 기존 자료(원래 research 폴더에 있던 파일과 SHA-256 일치 확인). KVA가 없는 번갈아뛰기 04.mp4 제외.",
               "- KVA Timestamp/AverageTimeStampsPerFrame으로 원본 프레임을 구한다. 점이 있는 고유 키프레임 하나를 점프 1회로 보고 연필이 여러 개여도 1회로 센다. 빈 주석 제외. 독립 수동 재주석은 수행하지 않았다.",
               "- 512×288 입력을 새 경로는 60Hz로 추론하고 신호 변환 전에 20Hz로 선택한다. 이전 CPU는 20Hz 입력이다. 원본이 30/24fps이면 60Hz 시각에서 가까운 원본 프레임을 반복 사용하며 이는 새 고유 영상 정보가 아니다. 첫 정답 0.5초 전부터 마지막 정답 0.5초 후까지 계수하고 앞선 프레임은 엔진 준비에 쓴다. 양쪽 모두 준비 게이트·카운트다운을 우회해 같은 라벨 구간을 비교했다. 인증·WebSocket·실시간 시작/종료는 별도 브라우저 테스트에서 확인한다.",
               "- 기존 JPEG 품질 62의 CPU 경로와 JPEG 없는 새 GPU 경로의 비교다. API·전처리·평활화·수치 연산 차이도 포함된다. GPU 하드웨어만의 인과 효과를 분리한 실험은 아니다.",
               "- 카운팅 상태기계·임계값·이중뛰기 분류기 가중치는 바꾸지 않았다. 60fps 결과의 CPU 비교군은 동일 해시의 영상·엔진으로 먼저 실측한 20fps CPU 결과를 재사용했으며, protocol.json의 reused_legacy에 기록했다. 영상별 개선은 보장되지 않으며 CSV에 모든 오차를 기록했다.",
               "- 기존 분류기 학습 자료와 평가 영상의 중복 여부는 알 수 없다. 새 사용자 일반화·독립 평가 정확도로 해석하지 않는다.",
               "", "## 재현", "",
               "Python 3.11 + requirements-dev.txt와 Chrome이 필요하다. 원본 영상·라벨은 수정하지 않았다. 아래 순서로 실행한다.", "",
               "1. python tools/fetch_pose_assets.py", "2. npm.cmd install --prefix .benchmark-node playwright@1.58.2",
               "3. 각 종목 원본 영상/라벨을 benchmark_results/cache/basic, alternating, double에 준비",
               "4. python tools/compare_pose_backends.py (로컬 서버 실행 유지)",
               "5. 별도 터미널: node tools/compare_pose_backends.cjs",
               "6. python tools/summarize_pose_comparison.py", "",
               "20fps 폴더의 초기 GPU/CPU 비교도 보존한다. --resume 옵션은 중단된 브라우저 비교를 이어간다. comparison.json은 원시 프레임 시간·계수 이벤트, summary.json은 집계, comparison.csv는 영상별 결과, protocol.json은 조건·소스 해시이다. 재실행하면 결과가 덮어써진다."]
    fps_path = OUT / "live-fps.json"
    if fps_path.exists():
        fps = json.loads(fps_path.read_text(encoding="utf-8"))
        report += ["", "## 실제 브라우저 FPS", "",
                   "60fps 원본 영상을 가짜 웹캠으로 재생하고 실제 앱의 GPU Worker → 인증 WebSocket → 서버 카운팅 → 화면 갱신을 실행했다. 각 종목 10초 세션 중 경과 2~9초 구간을 측정했다. 스크린샷은 측정 구간 전에 촬영했다. 물리 웹캠·원격 네트워크의 성능 측정은 아니다.", "",
                   "| 종목 | 설정 FPS | 실제 분석 FPS | 카메라 콜백 FPS | 도착 시 이전 처리 중이던 횟수 |",
                   "|---|---:|---:|---:|---:|"]
        for r in fps["results"]:
            report.append(f"| {NAMES[r['mode']]} | {r['settings']['frameRate']} | {r['analysisFps']:.2f} | {r['cameraCallbackFps']:.2f} | {r['busyCameraFrames']} |")
        report += ["", "중복 영상을 분석한 숫자를 더하지 않고 고유 카메라 프레임 기준 FPS도 확인했다(분석 FPS와 일치). 이 실행에서는 카메라 콜백 도착률 자체가 약 50fps였고, 프레임 도착 시 이전 분석·서버 응답을 기다리는 경우는 거의 없었고 도착률과 분석률이 비슷했다. 따라서 60fps 목표 설정은 완료했지만 이 Headless Chrome 재생 환경에서 실제 60fps를 달성했다고 주장하지 않는다.",
                   "", "정확도 비교는 고정 60Hz 타임스탬프로 입력을 모사한 별도 실험이다. 세 종목 FPS 측정에는 같은 60fps 영상을 사용해 코드 경로별 처리량을 비교했다. 세 종목 고유 영상의 정확도는 위 20개 라벨 영상 비교를 참조한다."]
        result["live_fps"] = [{key: r[key] for key in ("mode", "analysisFps", "cameraCallbackFps", "busyCameraFrames")} for r in fps["results"]]
        (OUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
