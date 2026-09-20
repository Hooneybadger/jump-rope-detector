---
name: "헤아리오 (Hearalo)"
description: "줄의 궤적과 계수 눈금으로 실시간 줄넘기 측정을 명료하게 운영하는 도구형 인터페이스"
colors:
  night: "#07111f"
  navy: "#0d1c2f"
  blue: "#1268ff"
  blue-dark: "#084fc7"
  lime: "#c8ff3d"
  coral: "#ff6b57"
  cyan: "#52d6d0"
  paper: "#f3f6f8"
  white: "#ffffff"
  ink: "#0a1728"
  muted: "#59677a"
  line: "#d9e0e7"
typography:
  display:
    fontFamily: "Gothic A1, Malgun Gothic, sans-serif"
    fontSize: "clamp(2.7rem, 4.4vw, 4.9rem)"
    fontWeight: 700
    lineHeight: 1.09
    letterSpacing: "-0.04em"
  headline:
    fontFamily: "Gothic A1, Malgun Gothic, sans-serif"
    fontSize: "clamp(2.45rem, 4.3vw, 4.35rem)"
    fontWeight: 700
    lineHeight: 1.08
    letterSpacing: "-0.04em"
  title:
    fontFamily: "Gothic A1, Malgun Gothic, sans-serif"
    fontSize: "1.85rem"
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: "-0.04em"
  body:
    fontFamily: "Gothic A1, Malgun Gothic, sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.7
    letterSpacing: "normal"
  label:
    fontFamily: "Gothic A1, Malgun Gothic, sans-serif"
    fontSize: "0.78rem"
    fontWeight: 700
    lineHeight: 1.4
    letterSpacing: "0.08em"
rounded:
  task: "3px"
  container: "4px"
  personal: "12px"
  avatar: "16px"
  pill: "999px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "16px"
  lg: "24px"
  xl: "32px"
  section: "48px"
  section-wide: "88px"
components:
  button-primary:
    backgroundColor: "{colors.blue}"
    textColor: "{colors.white}"
    typography: "{typography.label}"
    rounded: "{rounded.task}"
    padding: "15px 22px"
  button-primary-hover:
    backgroundColor: "{colors.blue-dark}"
    textColor: "{colors.white}"
    rounded: "{rounded.task}"
  button-secondary:
    backgroundColor: "{colors.white}"
    textColor: "{colors.ink}"
    typography: "{typography.label}"
    rounded: "{rounded.task}"
    padding: "15px 22px"
  field:
    backgroundColor: "{colors.white}"
    textColor: "{colors.ink}"
    typography: "{typography.body}"
    rounded: "{rounded.task}"
    padding: "14px 15px"
  mode-card:
    backgroundColor: "{colors.white}"
    textColor: "{colors.ink}"
    rounded: "{rounded.container}"
    padding: "28px"
  dialog-task:
    backgroundColor: "{colors.white}"
    textColor: "{colors.ink}"
    rounded: "{rounded.task}"
    padding: "36px"
  dialog-personal:
    backgroundColor: "{colors.white}"
    textColor: "{colors.ink}"
    rounded: "{rounded.personal}"
    padding: "32px"
---

# Design System: 헤아리오 (Hearalo)

## Overview

**Creative North Star: "The Measured Arc"**

헤아리오는 운동 계측기의 명료함을 줄의 궤적과 계수 눈금이라는 한 가지 시각 문법으로 묶는다. 화면은 장식보다 현재 상태, 다음 행동, 측정 결과를 먼저 드러내며, 네이비 계측 화면과 밝은 운영 화면 사이에서도 같은 리듬과 브랜드 표식을 유지한다.

밀도는 목적에 따라 달라진다. 대시보드와 관리 화면은 빠르게 훑을 수 있는 촘촘한 구조를 사용하고, 로그인·결과·개인 정보처럼 집중이나 확인이 필요한 순간에는 여백과 큰 숫자를 늘린다. 로고의 줄 호와 세 개의 계수 눈금, 종목 카드의 발동작 도해, 측정 패널의 반복 눈금이 이 시스템의 재사용 가능한 서명이다.

**Key Characteristics:**

- 계측기처럼 명료한 정보 위계와 큰 수치 표시
- 줄의 호, 계수 눈금, 발동작 도해로 이어지는 일관된 스포츠 문법
- 밝은 운영 면과 Night/Navy 실시간 측정 면의 선명한 대비
- 3–4px 작업형 모서리와 12–16px 개인·확인형 모서리의 목적별 구분
- 상태 변화에만 쓰는 180–200ms 피드백과 축소 모션 지원

## Colors

차가운 Night와 Paper가 작업 환경을 만들고, Blue는 행동, Lime은 실시간 상태, Coral은 경고와 초점을 짧고 명확하게 표시한다.

### Primary

- **Instrument Blue:** 주요 실행 버튼, 활성 상태, 기본 종목의 궤적에 사용한다.
- **Deep Action Blue:** Blue 버튼의 hover 상태에만 사용해 행동 계층을 유지한다.

### Secondary

- **Counting Lime:** 라이브 상태점, 시간 진행, 계수 눈금, 어두운 면의 브랜드 호에 사용한다.
- **Motion Cyan:** 번갈아뛰기 발동작과 궤적을 구분하는 종목 전용 색이다.

### Tertiary

- **Handle Coral:** 줄 손잡이, 파괴적 상태, 전역 키보드 초점 링에 사용한다.

### Neutral

- **Measurement Night:** 로그인 브랜드 면과 전체 화면 측정 배경이다.
- **Panel Navy:** 카운터 패널처럼 Night 위에 놓이는 계측 표면이다.
- **Cool Paper:** 앱 바깥 바탕과 약한 그룹 배경이다.
- **Clear White:** 카드, 표, 폼, 다이얼로그의 작업 표면이다.
- **Technical Ink:** 밝은 면의 본문과 핵심 수치다.
- **Operational Muted:** 보조 설명과 메타데이터다.
- **Rule Line:** 카드, 표 행, 필드 경계를 만드는 얇은 구조선이다.

### Named Rules

**The Signal Color Rule.** Blue는 행동, Lime은 실시간·계수, Coral은 초점·경고에만 배정하며 서로의 역할을 바꾸지 않는다.

**The Contrast Contract Rule.** 일반 텍스트는 배경 대비 4.5:1 이상을 유지한다. 현재 핵심 조합인 Muted/Paper, Blue/White, Lime/Night는 이 기준을 충족한다.

## Typography

**Display Font:** Gothic A1 (with Malgun Gothic and sans-serif fallback)  
**Body Font:** Gothic A1 (with Malgun Gothic and sans-serif fallback)

**Character:** 한글 획이 안정적인 단일 고딕 패밀리를 400과 700 두 무게로만 운용한다. 큰 제목과 숫자는 촘촘한 자간과 짧은 행간으로 계측기의 즉시성을 만들고, 설명문은 넉넉한 행간으로 읽기 부담을 낮춘다.

### Hierarchy

- **Display** (700, fluid display scale, 1.09): 로그인 메시지처럼 브랜드를 소개하는 가장 큰 문장에만 사용한다.
- **Headline** (700, fluid headline scale, 1.08): 대시보드와 관리자 화면의 첫 과업 제목에 사용한다.
- **Title** (700, compact title scale, 1.2): 섹션과 다이얼로그 제목에 사용한다.
- **Body** (400, base scale, 1.7): 안내, 설명, 폼 가이드에 사용하며 긴 문장에는 여유 있는 행간을 유지한다.
- **Label** (700, compact label scale, 0.08em): 상태, 종목 인덱스, 표 헤더처럼 짧고 반복되는 정보에 사용한다.
- **Live Count** (700, fluid 6rem–12rem, 0.8): 측정 중 점프 수를 시야의 최우선에 둔다.

### Named Rules

**The Two-Weight Rule.** Gothic A1의 400과 700만 사용한다. 강조는 새로운 글꼴이나 중간 무게가 아니라 크기, 색, 배치로 만든다.

**The Number Leads Rule.** 측정 화면에서는 횟수와 남은 시간이 주변 레이블보다 먼저 읽혀야 한다.

## Layout

데스크톱 대시보드는 최대 1500px의 중앙 컨테이너 안에서 3열 종목 카드, 전체 너비 기록 표, 2열 관리자 패널을 사용한다. 상단 내비게이션은 72px 높이로 고정되고, 주요 섹션은 48–88px 간격으로 분리한다. 작업 내부 간격은 8px 배수를 기본으로 하되 작은 상태·선택 제어에는 3–7px 단위의 촘촘한 간격을 허용한다.

1000px 이하에서는 종목 카드와 관리자 패널이 한 열로 바뀌고 주 내비게이션은 58px 높이의 하단 바로 이동한다. 측정 패널은 카메라 위 반투명 패널로 겹친다. 680px 이하에서는 대시보드 여백을 줄이고 종목 카드를 세로형으로 되돌리며, 측정 화면을 카메라 55dvh와 계수 패널 45dvh로 쌓는다. 결과 카드와 인증 필드도 한 열이 된다. 2200px 이상 측정 화면에서는 바와 패널, 버튼을 확장한다.

**The Task-First Responsive Rule.** 모바일에서는 브랜드 장식과 보조 요약을 먼저 줄이고, 종목 선택·측정 수치·종료 행동을 남긴다.

## Elevation & Depth

기본 구조는 선과 색면으로 구분하는 평면 시스템이다. 깊이는 다이얼로그, 결과 오버레이, hover 중인 종목 카드처럼 현재 상호작용이 다른 면 위로 올라오는 순간에만 넓고 부드러운 Night 계열 그림자를 사용한다. 측정 화면의 태블릿 패널은 Navy 반투명 면과 8px blur로 카메라 위에 겹친다.

### Shadow Vocabulary

- **Operational Lift** (`0 28px 70px rgba(7,17,31,.16)`): 다이얼로그, 결과 카드, hover 종목 카드에만 사용한다.
- **Live Signal Halo** (`0 0 0 6px rgba(200,255,61,.12)`): 실시간 상태점 주위에만 사용한다.

### Named Rules

**The Flat-at-Rest Rule.** 정지 상태의 카드와 패널은 테두리와 색면으로 구분하고, 그림자는 활성 오버레이나 직접적인 hover 피드백에만 사용한다.

## Shapes

작업 화면의 버튼, 입력, 카드, 표, 설정·기록 다이얼로그는 거의 직선에 가까운 3–4px 모서리를 사용한다. 프로필, 사용자 배지, 삭제 확인처럼 개인성이나 최종 확인이 있는 요소는 12–16px로 부드럽게 구분한다. 상태 칩과 작은 시작 배지는 완전한 pill이고, 로고와 시각 도해의 곡선은 줄의 호를 따른다.

선은 대부분 1px Rule Line이며, 라이브 카메라 가이드는 점선과 열린 모서리로 프레임을 암시한다. 종목 도해는 가는 줄 궤적, 굵은 손잡이, 발 모양, 점선 동작선을 조합한다.

**The Radius Means Context Rule.** 작은 반경은 작업, 큰 반경은 개인·확인 맥락이다. 장식 목적으로 반경을 섞지 않는다.

## Components

### Buttons

- **Shape:** 작업 버튼은 단단한 3px 모서리를 사용하고, 기본 내부 여백은 15px × 22px이다.
- **Primary:** Instrument Blue 면에 흰색 굵은 글자를 사용한다. hover는 Deep Action Blue, active는 1px 아래로 이동한다.
- **Secondary:** 흰색 면과 Rule Line 테두리, Technical Ink 글자를 사용한다.
- **Danger:** 흰색 경고 외곽선 또는 진한 Coral 계열 채움으로 삭제의 위험도를 구분한다.
- **Focus / Motion:** 모든 버튼은 3px Coral `focus-visible` 링을 가지며, 상태 전환은 180ms이다. 아이콘 전용 버튼은 최소 44×44px이다.

### Chips

- **Style:** 시간 프리셋은 1px Rule Line과 흰색 면을 사용하고 선택되면 Blue 면과 흰 글자로 바뀐다. 프로필 권한은 옅은 Blue pill로 표현한다.
- **State:** 선택은 색과 테두리를 함께 바꾸며 색만으로 의존하지 않는다.

### Cards / Containers

- **Corner Style:** 종목 카드와 표 컨테이너는 4px 이하의 작은 모서리다.
- **Background:** 밝은 작업 면은 White, 앱 배경은 Paper다.
- **Shadow Strategy:** 정지 상태에서는 그림자가 없고 종목 카드 hover에서만 Operational Lift를 사용한다.
- **Border:** 1px Rule Line이 카드와 행의 구조를 만든다.
- **Internal Padding:** 종목 카드 28px, 관리자 패널 26px, 결과 영역 52px을 기준으로 목적에 맞게 조절한다.

### Inputs / Fields

- **Style:** 흰색 면, 1px 중립 테두리, 3px 모서리, 14px × 15px 내부 여백을 사용한다.
- **Focus:** 전역 3px Coral 링과 3px 바깥 여백을 사용한다.
- **Error / Disabled:** 오류는 짧은 적색 상태문으로 필드 가까이에 표시하고, 비활성 제어는 불투명도를 낮추며 커서를 바꾼다.

### Navigation

72px 상단 바는 로고, 가운데 탭, 사용자 클러스터를 한 줄에 둔다. 활성 탭은 Technical Ink 글자와 4px Counting Lime 표시선으로 드러난다. 1000px 이하에서는 동일한 탭을 58px 하단 바로 이동하며 활성 표시선은 위쪽에 놓는다.

### Mode Card

각 종목 카드는 줄의 궤적, 손잡이, 발동작, 동작선을 SVG로 보여주고 종목별 accent를 Blue, Cyan, Coral로 구분한다. 시작 행동은 카드 아래의 구조선과 작은 pill 배지로 연결한다. 권한이 없는 카드는 흐리게 만들고 시작 행동을 설명문으로 교체한다.

### Live Counter

전체 화면 측정은 Night 카메라 면과 Navy 계수 패널로 분리한다. 점프 수는 가장 큰 숫자, 남은 시간은 두 번째, 연결·신체 인식 상태는 작은 레이블로 배치한다. 시간 바는 Lime으로 200ms 선형 갱신하고, 반복 눈금은 계수 리듬을 시각화한다.

### Dialogs

설정·계정·기록 다이얼로그는 3–4px 작업형 모서리, 프로필과 삭제 확인은 12px 개인·확인형 모서리를 사용한다. 모든 네이티브 `dialog`는 제목과 `aria-labelledby`로 연결하고, 설명이 필수인 확인창은 `aria-describedby`도 연결한다.

## Do's and Don'ts

### Do:

- **Do** 줄의 호, 계수 눈금, 발동작 도해를 브랜드와 기능이 만나는 지점에 재사용한다.
- **Do** Blue, Lime, Coral의 역할을 고정하고 모든 일반 텍스트에서 최소 4.5:1 대비를 유지한다.
- **Do** 작업 제어에는 3–4px, 개인·확인 맥락에는 12–16px 반경을 사용한다.
- **Do** 아이콘 전용 상호작용에 최소 44×44px 목표 영역과 명시적 접근성 이름을 제공한다.
- **Do** 180–200ms 상태 피드백만 사용하고 `prefers-reduced-motion`에서 전환을 제거한다.
- **Do** 모바일에서 하단 내비게이션과 우선 과업을 먼저 보존한다.

### Don't:

- **Don't** 줄넘기와 무관한 인체 실루엣이나 일반 스포츠 아이콘으로 모드 도해를 대체하지 않는다.
- **Don't** 정지 카드에 그림자를 쌓거나 작업 화면 전체를 둥근 카드 모음으로 만들지 않는다.
- **Don't** Blue, Lime, Coral을 장식용으로 넓게 칠해 상태 의미를 희석하지 않는다.
- **Don't** 자동 재생, 반복, 스프링 모션을 추가하지 않는다.
- **Don't** 좁은 화면에서 측정 수치나 종료 행동을 보조 설명보다 먼저 숨기지 않는다.
