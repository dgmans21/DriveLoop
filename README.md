# DriveLoop

CARLA 시뮬레이터에서 **데이터 수집 → 자동 라벨링 → 품질관리·중복 제거 → 학습 → 실패 분석 → 표적 재수집**까지
자율주행 인지 데이터 파이프라인을 직접 만들고, 그렇게 학습한 내 모델의 인지만으로 신호등을 지키며 주행하는 프로젝트.

**결과 페이지**: https://dgmans21.github.io/DriveLoop/ (주행 데모 · 2D 지도 · 데이터 개선 과정, 한국어/English)

## 핵심 결과

| 항목 | 결과 |
|---|---|
| 처음 보는 도시(Town05) 주행, 4개 날씨 × 180초 | **빨간불 위반 0회**, 고른 신호의 색 정확도 99.6~100% |
| 데이터만 바꾼 효과 (학습 맵 2개 → 4개, 같은 학습 설정·같은 test 815장) | 빨간불 재현율 0.764 → **0.815**, 노란불 0.733 → **0.796**, 차량 mAP50 0.765 → 0.813 |
| 먼 신호·작은 객체 | 빨간불 12–16px 0.62 → 0.74, 차량 12–16px 0.46 → 0.59 |
| 추론 | YOLO11n 1280px, 중앙값 약 12ms (RTX 4060 Ti 8GB) |
| 정답값 인지 vs 내 모델 인지 (같은 경로·신호, 40회) | 정상 정지 52/52 vs 52/52, 위반 0 vs 0, 판단 일치율 97.6%, 출발 지연 0.15초. 비교로 위험 장면 2건 발견 (판단 보류 중 늦은 제동, 멈출 수 없는 거리 급제동) |

## 설계 원칙

- **인지 / 판단 / 제어 분리**: 판단(`planning`)은 `PerceptionOutput`만 입력으로 받는다.
  1단계는 시뮬레이터 정답값(`GroundTruthPerception`), 2단계는 내 모델(`ModelPerception`)이 같은 형식을 채운다.
  판단·제어 코드는 바꾸지 않으므로 3단계에서 같은 조건으로 공정하게 비교할 수 있다.
- **위치는 지도, 색은 모델**: 지도로 내 차선 신호등의 3D 위치를 화면에 투영하고, 그 근처 검출만 내 신호로 채택한다.
  최근 5프레임 투표로 안정화하고, 안전 순서 RED > YELLOW > GREEN으로 판단한다.
- **재현 가능한 데이터**: 수집 조건은 YAML, 시드는 에피소드 ID에서 결정, 처리 단계는 DVC 파이프라인.
- **평가는 처음 보는 도시로**: Town05는 학습·검증에 쓰지 않고 test 전용으로 둔다.
- **판단·제어는 CARLA 없이 테스트**: `behavior`, `pid`, `lateral`, `association` 등은 `pytest`로 검증한다 (103개).
- **8GB VRAM 운용**: 시뮬레이션과 학습은 동시에 돌리지 않는다. 학습은 10 epoch 단위로 끊어 재개한다.

## 디렉터리 구조

```
DriveLoop/
├─ configs/
│  ├─ sim.yaml, driving.yaml     # 서버·맵·dt / 속도·PID·신호 판단·pure pursuit
│  ├─ collection/                # 수집 매트릭스 (맵 × 날씨 × 시간 × 교통량), v1·v2·v3
│  ├─ labeling.yaml, qc.yaml, dedup.yaml, report.yaml
│  ├─ export/                    # 데이터셋 구성 (원본 조합, val 에피소드, test 맵)
│  └─ train/                     # 학습 설정 (v2_base, v3_base: 데이터만 다르고 설정 동일)
├─ src/driveloop/
│  ├─ sim/                       # CARLA 접속, 동기 모드, 센서, 날씨, NPC
│  ├─ perception/
│  │  ├─ types.py                # PerceptionOutput: 인지 ↔ 판단 인터페이스
│  │  ├─ ground_truth.py         # 정답값 인지 + 지도 기반 '내 신호등' 찾기
│  │  ├─ association.py          # 예상 위치 ↔ 검출 매칭, 시간 투표 (RED > YELLOW 우선)
│  │  └─ model.py                # YOLO 기반 내 모델 인지
│  ├─ planning/                  # 경로, 신호 상태 머신, 커브 속도 제한
│  ├─ control/                   # 속도 PID, pure pursuit (후륜축 기준)
│  ├─ data/                      # 수집, 자동 라벨링, 투영, 메타데이터·QC, 중복 제거, 내보내기
│  ├─ eval/detection.py          # IoU 매칭, 조건·크기별 재현율, 색 혼동
│  └─ viz/                       # pygame HUD, 오버레이·MP4 녹화
├─ scripts/                      # 실행 진입점 (단계별 번호)
├─ tests/                        # CARLA 비의존 단위 테스트
├─ docs/                         # 결과 페이지 코드 (index.html, assets/*.js·css) + SETUP.md
├─ data/                         # DVC로 관리 (git 제외)
└─ dvc.yaml, dvc.lock
```

## 로드맵

| 단계 | 내용 | 상태 |
|---|---|---|
| 1 | 자차 스폰, 카메라, 정답값 인지로 신호등 주행 (상태 머신, 노란불 딜레마, PID, pure pursuit, 커브 감속) | ✅ `01`~`03` |
| 2-1 | 수집 매트릭스(YAML) → 배치 수집 (RGB, 인스턴스 세그, 3D 정답) | ✅ `10_collect.py` |
| 2-2 | 자동 라벨링 → 메타데이터·QC → 중복 제거 → YOLO 내보내기 → 분포 리포트 (DVC 파이프라인) | ✅ `20`~`25` |
| 데이터 v1→v2 | 리포트 부족 목록 기반 보강 (노란불 6초 + 조건부 저장, Town05 test 전용) | ✅ |
| 2-3 | YOLO11n 학습 (10 epoch 단위 재개), 조건·크기별 평가, 신뢰도 임계값 스윕 | ✅ `30`~`32` |
| 데이터 v3 | 일반화 갭(val 0.93 vs test 0.76) → 학습 맵 2개 추가 (Town02·04), 같은 설정으로 재학습 | ✅ 빨간불 재현율 +5.1%p |
| 2-4 | 내 모델로 인지 교체 주행, 날씨·시간 옵션, 웹용 녹화(영상 + 프레임별 JSON) | ✅ `04_model_drive.py` |
| 결과 페이지 | 오버레이 플레이어 + 2D 지도 + 타임라인, 한국어/영어, gh-pages 단일 커밋 배포 | ✅ `40_build_site.py`, `41_publish_site.ps1` |
| 3 | 정답값 인지 vs 모델 인지 주행 비교 (Town05, 4개 날씨 × 시드 5 × 2 = 40회). 판단·제어는 `agent.py`로 동일 | ✅ 정지해야 할 52번 모두 정답값과 같이 정지, 위반 0. 위험 장면 2건 발견 → 3+ |
| 3+ | 안전장치: 정지선 가까이에서 신호 판단 보류(UNKNOWN)면 감속 (지금은 직전 판단 유지 → 끝까지 못 보면 통과 위험) | 📝 다음 |
| 1 미완 | **장애물(앞차) 대응** — 처음 범위에 있었으나 신호등만 구현. 차량은 검출·평가까지만 있고 주행 판단에 안 씀 | 📝 다음 |
| ACC | 앞차 따라가기: ① 판단(시간 간격 유지, 앞차 정지 시 정지) + 앞차 거리는 정답값 → ② 모델 차량 검출 + 카메라 거리 추정으로 교체, 정답값 대비 비교 → 데모·비교에 NPC 추가 | 📝 |
| v4 | 보행자: 수집 → 클래스 추가 → 재학습 → "앞에 있으면 정지" (ACC 정지 로직 재사용) | 📝 |
| 최종 검증 | 차량·보행자가 있는 환경에서 시드 확대 + 시나리오(노란불 딜레마, 먼 신호, 비·밤) — 빈 도로 검증을 먼저 키우지 않는 이유: 시스템이 바뀌면 다시 해야 함 | 📝 |
| 그 외 | sim-to-real (실제 도로 데이터), 눈 날씨 (CARLA 0.9.16 미지원) | 📝 |

## 빠른 실행

설치는 [docs/SETUP.md](docs/SETUP.md)를 따른다.

```powershell
conda activate driveloop
powershell -ExecutionPolicy Bypass -File scripts\start_carla.ps1 -Quality Epic -OffScreen   # CARLA 서버
python scripts\00_check_env.py                 # 접속·버전 확인 (--reset: 동기 모드 해제)
python scripts\03_traffic_light_drive.py       # 1단계: 정답값 인지 주행

# 데이터 → 학습 → 평가
python scripts\10_collect.py --config configs\collection\v3.yaml --limit 1   # 파일럿
dvc repro                                      # 라벨 → QC → 중복 제거 → 내보내기 → 리포트
python scripts\30_train.py --config configs\train\v3_base.yaml --stop-after 10   # 10 epoch씩
python scripts\30_train.py --config configs\train\v3_base.yaml --resume --stop-after 10
python scripts\31_evaluate.py --run v3_base --dataset v3 --splits test --conf 0.10

# 내 모델로 주행 (처음 보는 도시, 날씨 지정, 웹용 녹화)
python scripts\04_model_drive.py --map Town05 --weather rain --time night --seconds 180 --record --web --no-display

# 3단계: 정답값 vs 모델 주행 비교 (끝난 실행은 건너뜀) → 분석
python scripts\50_compare_drive.py --only clear_noon,rain_noon
python scripts\51_compare_report.py

# 결과 페이지
python scripts\40_build_site.py                # outputs/web → docs/assets (영상 압축, 2D 지도)
python -m http.server 8765 --directory docs    # 로컬 미리보기 http://localhost:8765
powershell -ExecutionPolicy Bypass -File scripts\41_publish_site.ps1   # gh-pages = 커밋 1개로 교체
git push -f origin gh-pages

pytest                                         # CARLA 없이 단위 테스트
```

## 데이터 버전 관리 (DVC)

원본 `data/raw/v1~v3`은 `dvc add`로, 그 뒤 단계는 [dvc.yaml](dvc.yaml) 파이프라인으로 관리한다.
git에는 데이터의 지문(`*.dvc`, `dvc.lock`)과 품질 리포트만 들어가고, 실제 파일은 DVC 저장소에 있다.
ultralytics가 데이터셋 폴더에 쓰는 라벨 캐시는 `.dvcignore`로 해시에서 제외한다.

```powershell
dvc pull                 # DVC 저장소에서 데이터 받기 (기본: 로컬 D:\dvc-store)
dvc repro                # 규칙·코드가 바뀐 단계만 다시 실행
dvc metrics show         # QC·중복 제거 지표
dvc push                 # 새 결과를 DVC 저장소에 백업
```

## 결과 페이지 배포

`docs/`에는 페이지 코드만 커밋하고, 영상·프레임 JSON 같은 빌드 결과물은 `.gitignore`로 뺀다.
`41_publish_site.ps1`이 `docs/` 전체를 **부모 없는 커밋 1개**로 `gh-pages`에 덮어쓰므로,
데모 영상을 여러 번 교체해도 저장소 기록이 커지지 않는다. GitHub Pages 설정: Branch `gh-pages`, 폴더 `/`.
