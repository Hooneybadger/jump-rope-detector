# Jump Rope Lab

MediaPipe Pose 기반 실시간 줄넘기 측정 웹 애플리케이션입니다. 기존 프로젝트의 검증된 모아뛰기·번갈아뛰기·이중뛰기 카운팅 엔진은 유지하고, 화면·인증·권한·데이터 저장·배포 구조를 새로 만들었습니다.

브라우저가 PC 웹캠을 직접 열기 때문에 Docker 컨테이너에 카메라 장치를 연결할 필요가 없습니다. 촬영 프레임은 로그인된 브라우저와 서버 사이의 WebSocket으로 전달되며 저장되지 않습니다. 서버는 MediaPipe Pose와 선택된 카운팅 엔진으로 프레임을 처리하고, 관절 오버레이와 카운트 상태만 브라우저에 반환합니다.

## 주요 기능

- 모아뛰기, 번갈아뛰기, 이중뛰기 실시간 측정
- 전신 인식 후 준비 1초와 카운트다운 3초를 거쳐 자동 시작
- 데스크톱·태블릿·모바일 반응형 화면과 보통/넓게/전체 화면 전환
- 로그인, 계정 잠금, DB 세션, 관리자/일반 사용자 역할
- 사용자별 종목 사용 권한 및 기록 조회 권한
- 사용자별 측정 횟수·시간·종목 이력 저장
- 사용자 생성·권한 변경·로그인·측정 시작/완료 감사 로그
- PostgreSQL 영속 저장, Alembic 스키마 마이그레이션
- Nginx, FastAPI, PostgreSQL로 구성된 Docker Compose 실행 환경

## 실행 요구사항

- Docker Desktop 또는 Docker Engine + Docker Compose v2
- 웹캠이 연결된 PC
- Chrome, Edge 등 `getUserMedia`와 WebSocket을 지원하는 최신 브라우저

로컬의 `http://localhost`는 브라우저가 카메라 사용을 허용하는 보안 컨텍스트로 취급합니다. 다른 PC나 도메인에서 접속할 때는 반드시 HTTPS를 적용해야 합니다.

## 빠른 실행

git bash에서 프로젝트 디렉터리로 이동합니다.

```git bash
cd jump-rope-detector
cp .env.example .env
```

`.env`를 열어 아래 두 값을 반드시 변경합니다.

```dotenv
POSTGRES_PASSWORD=충분히_긴_무작위_DB_비밀번호
ADMIN_PASSWORD=12자_이상의_관리자_비밀번호
```

비밀번호는 코드나 Git에 커밋하지 마세요. DB 접속 문자열을 직접 조합하지 않고 개별 환경 변수로 전달하므로 `@`, `:`, `/` 같은 특수문자도 사용할 수 있습니다.

서비스를 빌드하고 시작합니다.

```powershell
docker compose up --build -d
docker compose ps
```

브라우저에서 [http://localhost:8080](http://localhost:8080)을 열고 `.env`의 `ADMIN_USERNAME`과 `ADMIN_PASSWORD`로 로그인합니다. 최초 실행 시 DB 마이그레이션과 관리자 계정 생성이 자동 수행됩니다.

로그 확인과 종료 방법은 다음과 같습니다.

```powershell
docker compose logs -f app
docker compose down
```

`docker compose down`은 DB 볼륨을 보존합니다. 아래 명령은 모든 사용자와 측정 기록을 포함한 DB 볼륨을 삭제하므로 초기화가 정말 필요한 경우에만 사용하세요.

```powershell
docker compose down -v
```

## 최초 설정

관리자 로그인 후 상단의 **관리** 메뉴에서 사용자를 추가합니다.

1. **사용자 추가**를 누릅니다.
2. 아이디, 표시 이름, 12자 이상의 임시 비밀번호를 입력합니다.
3. 모아뛰기, 번갈아뛰기, 이중뛰기, 기록 조회 권한을 선택합니다.
4. 생성 후 사용자 목록의 체크박스로 종목 권한을 즉시 변경할 수 있습니다.

관리자는 모든 종목과 전체 측정 기록을 볼 수 있습니다. 일반 사용자는 자신에게 부여된 종목만 시작할 수 있고, 기록 조회 권한이 있을 때만 자신의 최근 기록을 확인합니다.

## 측정 워크플로우

1. 로그인 후 훈련 화면에서 사용할 줄넘기 종목을 선택합니다.
2. 브라우저의 카메라 사용 요청을 허용합니다.
3. 카메라에서 머리부터 발끝까지 보이도록 약 2~3m 거리를 확보합니다.
4. 서버가 전신 자세를 1초간 안정적으로 인식하면 3초 카운트다운이 시작됩니다.
5. 선택한 카운팅 엔진이 프레임을 분석하고 횟수와 측정 시간을 실시간으로 갱신합니다.
6. **측정 끝내기**를 누르면 결과가 PostgreSQL에 저장됩니다.
7. **기록 확인**을 누르면 대시보드에서 최근 세션과 누적 기록을 확인할 수 있습니다.

세 종목은 다음 원본 알고리즘을 그대로 사용합니다.

| 종목 | 엔진 | 주요 판정 신호 |
|---|---|---|
| 모아뛰기 | `basic_jump/counter_engine.py` | 양발 접지, 골반 반동, 좌우 대칭 |
| 번갈아뛰기 | `alternating_jump/counter_engine.py` | 좌우 지지 발 전환, 발·무릎·손목 특징 |
| 이중뛰기 | `double_jump/counter_engine.py` | 체공 주기, 골반·발 움직임, 손목 회전 분류 |

## 시스템 구조

```text
웹캠
  ↓ 브라우저 getUserMedia
반응형 웹 UI
  ↓ 인증 쿠키 + Origin 검증 WebSocket/JPEG
Nginx :8080
  ↓
FastAPI :8000
  ├─ MediaPipe Pose
  ├─ 줄넘기 카운팅 엔진 3종
  ├─ 인증·RBAC·CSRF·감사 로그
  └─ Alembic 마이그레이션
       ↓
PostgreSQL 17 영속 볼륨
```

- `web`: 외부 요청과 WebSocket을 받는 Nginx 리버스 프록시
- `app`: FastAPI API, 정적 UI, MediaPipe 처리, 카운팅 엔진
- `db`: 사용자, 서버 세션, 운동 기록, 감사 로그를 저장하는 PostgreSQL

Streamlit과 `streamlit-webrtc`는 제거했습니다. UI 크기와 상태 관리, 인증/RBAC, DB 트랜잭션을 일관되게 제어하기 위해 브라우저 UI + FastAPI 구조로 교체했습니다. 카메라는 여전히 사용자의 브라우저에서만 열리며, 서버에는 영상 파일을 저장하지 않습니다.

## 환경 변수

| 변수 | 설명 |
|---|---|
| `POSTGRES_DB` | PostgreSQL DB 이름 |
| `POSTGRES_USER` | PostgreSQL 사용자 |
| `POSTGRES_PASSWORD` | PostgreSQL 비밀번호 |
| `ADMIN_USERNAME` | 최초 관리자 아이디 |
| `ADMIN_PASSWORD` | 최초 관리자 비밀번호, 최소 12자 |
| `ADMIN_DISPLAY_NAME` | 관리자 표시 이름 |
| `COOKIE_SECURE` | HTTPS 운영 환경에서는 `true` |
| `ALLOWED_HOSTS` | 허용할 Host 헤더 목록 |
| `ALLOWED_ORIGINS` | WebSocket 연결을 허용할 정확한 Origin 목록 |
| `MAX_CONCURRENT_STREAMS` | 동시에 실행할 실시간 측정 수, 기본 4 |

LAN이나 도메인으로 서비스할 경우 실제 호스트와 Origin을 모두 설정합니다.

```dotenv
COOKIE_SECURE=true
ALLOWED_HOSTS=jump.example.com
ALLOWED_ORIGINS=https://jump.example.com
```

현재 Nginx 설정은 로컬 HTTP용입니다. 운영 환경에서는 앞단 리버스 프록시 또는 로드 밸런서에서 TLS 인증서를 적용하고 WebSocket 업그레이드 헤더를 전달하세요.

## 보안 설계

- 비밀번호는 Argon2id로 해시하며 평문으로 저장하지 않습니다.
- 세션 쿠키는 `HttpOnly`, `SameSite=Strict`로 발급하고 운영 HTTPS에서는 `Secure`를 사용합니다.
- DB에는 세션 토큰 원문 대신 SHA-256 해시만 저장합니다.
- 상태 변경 API는 CSRF 토큰을 검증합니다.
- WebSocket은 로그인 세션, Origin, 사용자별 종목 권한을 연결 시점에 검증합니다.
- 로그인은 IP 단위 요청 제한과 계정 단위 5회 실패/15분 잠금을 적용합니다.
- CSP, 클릭재킹 방지, MIME 스니핑 방지, 카메라 권한 제한 헤더를 적용합니다.
- 이미지 크기와 해상도, 동시 스트림 수를 제한합니다.
- 앱 컨테이너는 비루트 사용자, 읽기 전용 파일시스템, Linux capability 제거, `no-new-privileges`로 실행합니다.
- PostgreSQL 포트는 호스트에 공개하지 않습니다.
- 관리자 본인의 계정을 비활성화하거나 일반 사용자로 낮추는 변경을 차단합니다.
- 런타임 이미지에서 패키지 설치 도구(`pip`, `setuptools`)를 제거해 공격 표면을 줄입니다.

운영에서는 `.env`를 비밀 저장소로 대체하고, DB 백업·TLS·로그 보존 정책을 별도로 구성하세요.

기존 카운팅 엔진이 사용하는 legacy `mp.solutions.pose` API는 MediaPipe 0.10.21과 protobuf 4.x 조합을 요구합니다. 현재 알려진 protobuf 취약점의 영향면을 줄이기 위해 애플리케이션은 protobuf 파일·메시지를 외부 입력으로 받지 않고, 크기 제한된 JPEG 프레임만 디코딩합니다. 향후 MediaPipe Tasks Pose Landmarker로 추출 계층을 이전하면 protobuf 6 이상으로 갱신할 수 있습니다.

## 개발 및 검증

로컬 Python 개발은 3.11을 권장합니다. MediaPipe 0.10.21의 플랫폼별 wheel 지원 범위 때문에 다른 Python 버전에서는 설치되지 않을 수 있습니다.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
pytest -q
ruff check app tests
node --check static/app.js
```

Docker 구성 검증과 이미지 빌드는 다음 명령으로 수행합니다.

```powershell
docker compose config --quiet
docker compose build app
```

## 문제 해결

### 카메라 요청이 나타나지 않음

- 주소가 `http://localhost:8080`인지 확인합니다.
- 브라우저 주소창의 사이트 권한에서 카메라를 허용합니다.
- 다른 앱이 웹캠을 점유하고 있지 않은지 확인합니다.
- 원격 접속이면 HTTP가 아니라 HTTPS를 사용합니다.

### 계속 “자세 찾는 중”으로 표시됨

- 머리부터 발끝까지 화면 안에 들어오도록 뒤로 이동합니다.
- 발과 바닥이 잘 구분되도록 조명을 밝게 합니다.
- 카메라를 허리보다 낮게 두지 말고 전신 정면을 촬영합니다.

### 로그인이 되지 않음

- `.env`의 관리자 아이디와 비밀번호를 확인합니다.
- 5회 실패했다면 15분 후 다시 시도합니다.
- 이미 생성된 DB 볼륨에서는 `.env`의 관리자 비밀번호 변경이 기존 계정 비밀번호를 덮어쓰지 않습니다.

### 컨테이너 상태가 healthy가 아님

```powershell
docker compose ps
docker compose logs app
docker compose logs db
```

DB 비밀번호나 마이그레이션 오류를 확인한 뒤 서비스를 다시 시작합니다.
