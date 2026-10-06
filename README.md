# DriveLoop

CARLA 시뮬레이터에서 신호등과 장애물에 반응하는 주행 데모를 만들고, 인지를 직접 학습한 모델로 교체하는 프로젝트.
인지 데이터를 수집·라벨링·검증·버전 관리하는 데이터 파이프라인이 핵심이다.

## 설계 원칙

- **인지 / 판단 / 제어 분리**: 판단(`planning`)은 `PerceptionOutput`만 입력으로 받는다.
  1단계는 시뮬레이터 정답값(`GroundTruthPerception`), 2단계는 내 모델이 같은 형식을 채운다.
  판단·제어 코드는 바꾸지 않으므로 3단계에서 같은 조건으로 공정하게 비교할 수 있다.
- **판단·제어는 CARLA 없이 테스트**: `behavior`, `pid`, `lateral`은 순수 Python이라 `pytest`로 검증한다.
- **설정은 YAML**: 맵, 속도, 게인, 수집 조건을 코드 밖에서 관리한다.
- **8GB VRAM 운용**: 저품질 모드, Town01~05, 낮은 해상도, 동기 모드를 쓴다. 시뮬레이션과 학습은 동시에 돌리지 않는다.

## 디렉터리 구조

```
DriveLoop/
├─ configs/
│  ├─ sim.yaml                 # 서버 접속, 맵, dt, 카메라
│  ├─ driving.yaml             # 속도, PID 게인, 신호 판단 파라미터
│  └─ collection/              # [2단계] 날씨×시간×맵×교통량 수집 매트릭스
├─ src/driveloop/
│  ├─ config.py                # YAML → dataclass (오타 키는 에러)
│  ├─ sim/                     # CARLA 접속, 동기 모드, 스폰, 센서
│  ├─ perception/
│  │  ├─ types.py              # PerceptionOutput: 인지 ↔ 판단 인터페이스
│  │  ├─ ground_truth.py       # [1단계] 시뮬레이터 정답값
│  │  └─ model.py              # [2단계] YOLO 기반 내 모델
│  ├─ planning/
│  │  ├─ route.py              # 차선 따라가는 waypoint 경로
│  │  └─ behavior.py           # 신호/장애물 상태 머신
│  ├─ control/
│  │  ├─ pid.py                # 종방향: 속도 PID → throttle/brake
│  │  └─ lateral.py            # 횡방향: pure pursuit
│  ├─ viz/hud.py               # pygame 화면 + 오버레이
│  ├─ data/                    # [2단계] 수집, 자동 라벨링, 메타데이터, QC, 중복 제거, 분포 분석
│  ├─ train/                   # [2단계] 학습, 조건별 평가, 실패 분석
│  ├─ eval/                    # [3단계] 시나리오 러너, 주행 지표 (신호 위반, 정지 성공률, 충돌)
│  └─ search/                  # [선택] 장면 임베딩 + 자연어 검색
├─ scripts/                    # 실행 진입점 (단계별 번호)
├─ tests/                      # CARLA 비의존 단위 테스트
├─ data/                       # [2단계] DVC로 관리 (git 제외)
└─ docs/SETUP.md               # 설치 가이드
```

`[n단계]`로 표시한 항목은 해당 단계에서 만든다.

## 로드맵

| 단계 | 내용 | 상태 |
|---|---|---|
| 1-1 | 자차 스폰 | ✅ `scripts/01_spawn_vehicle.py` |
| 1-2 | 카메라 화면 표시 (키보드 운전) | ✅ `scripts/02_camera_view.py` |
| 1-3 | 자율 주행 + 신호등 정지/출발, 노란불 판단, HUD | ✅ `scripts/03_traffic_light_drive.py` |
| 1-3a | 커브 감속 + 경로 기준 정지선 거리 | 🔧 코드·단위 테스트 완료, 실주행 확인은 2-4에서 |
| 2-1 | 수집 매트릭스(YAML) → 배치 원본 수집 (RGB, 인스턴스 세그, 3D 정답) | ✅ `scripts/10_collect.py`, v1: 16 에피소드 / 2,400장 / 0.45GB (Epic) |
| 2-2a | 자동 라벨링: 차량은 세그 인스턴스 기준(주차 차량 포함), 신호등은 3D 헤드 투영 + 정면 각도 필터 | ✅ `scripts/20_autolabel.py`, `21_preview_labels.py` (규칙 v3) |
| 2-2b | 메타데이터(frames/objects Parquet) + QC 플래그 + 신호등 불빛 색 검증 | ✅ `scripts/22_build_metadata.py` |
| 2-2c | 중복 제거 (자차 이동 + dHash + 라벨 구성, 대표 프레임 기준) | ✅ `scripts/23_dedup.py`, v1: 2,400 → 1,683장 |
| 2-2d | YOLO 내보내기: 에피소드 단위 분할, 무시 영역(회색) 처리, manifest | ✅ `scripts/24_export_yolo.py`, v1: train 1,199 / val 484장 |
| 2-2d+ | DVC: 원본 `dvc add` + 라벨→QC→중복 제거→내보내기 `dvc.yaml` 파이프라인 | ✅ `dvc repro`, `dvc metrics show` |
| 2-2e | 분포 리포트 (클래스·조건·신호등 크기·무시 사유, 목표 대비 부족 목록 자동 산출) | ✅ `scripts/25_distribution_report.py` → `data/reports/v1/distribution.html` |
| v2 | 리포트의 부족 목록 기반 추가 수집 (노란불 시간 6초 + 조건부 저장, Town05 평가 전용, 새 seed) | ✅ 원본 24 에피소드 / 4,154장 |
| 데이터셋 v2 | 원본 v1+v2 통합, train/val/test(Town05) | ✅ 3,987장, 부족 목록 10 → 3건 (노란불 train 285·val 78이 아직 목표 미달) |
| 시연 | pygame 화면 녹화(MP4) → 결과 웹 페이지(GitHub Pages) → (선택) 장면 검색 웹 앱 | 📝 2-4·3단계에서 |
| 2-3 | YOLO 학습, 조건별 성능, 실패 분석 → 추가 수집 | 🔧 `scripts/30_train.py`, 1 epoch 시험 학습 완료 (YOLO11n·1280px, VRAM 6.0GB, 약 2분/epoch) |
| 2-4 | 전방 차량 추종(정답값) + 내 모델로 인지 교체 | ⏳ |
| 2-4+ | 검토: 날씨 대응 판단 (비 → 순항 속도↓, 정지 감속도 기준↓, 앞차 간격↑). 이미지 날씨 분류기는 수집 태그를 라벨로 사용 | 📝 |
| 3+ | 검토: 비 시나리오에서 타이어 마찰을 낮춰 평가 (CARLA 날씨는 시각 효과만 있고 마찰은 바뀌지 않음) | 📝 |
| 2-3+ | 검토: 눈 합성 증강 확률 p 실험 (CARLA에는 눈 없음). 실제 눈 이미지로 효과 측정 | 📝 |
| 3 | 정답값 주행과 모델 주행 비교, sim-to-real 평가 | ⏳ |
| 선택 | 자연어 장면 검색 | ⏳ |

## 빠른 실행

설치는 [docs/SETUP.md](docs/SETUP.md)를 따른다.

```powershell
conda activate driveloop
powershell -ExecutionPolicy Bypass -File scripts\start_carla.ps1   # CARLA 서버 실행
python scripts\00_check_env.py          # 접속·버전 확인
python scripts\01_spawn_vehicle.py      # 1-1
python scripts\02_camera_view.py        # 1-2
python scripts\03_traffic_light_drive.py  # 1-3
python scripts\10_collect.py --dry-run  # 2-1 수집 계획 확인
python scripts\10_collect.py --limit 1  # 2-1 파일럿 (에피소드 1개)
pytest                                  # 판단·제어 단위 테스트 (CARLA 불필요)
```

## 데이터 버전 관리 (DVC)

원본 `data/raw/v1`은 `dvc add`로, 그 뒤 단계는 [dvc.yaml](dvc.yaml) 파이프라인으로 관리한다.
git에는 데이터의 지문(`*.dvc`, `dvc.lock`)과 품질 리포트만 들어가고, 실제 파일은 DVC 저장소에 있다.

```powershell
conda activate driveloop
dvc pull                 # DVC 저장소에서 데이터 받기 (기본: 로컬 D:\dvc-store)
dvc repro                # 규칙·코드가 바뀐 단계만 다시 실행 (라벨 → QC → 중복 제거 → 내보내기)
dvc metrics show         # QC·중복 제거 지표
dvc metrics diff HEAD~1  # 직전 커밋 대비 지표 변화
dvc push                 # 새 결과를 DVC 저장소에 백업
```
