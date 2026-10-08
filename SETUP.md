# 설치 및 환경 세팅 (Windows 11 + RTX 4060 Ti 8GB)

## 버전 선택

| 항목 | 선택 | 이유 |
|---|---|---|
| CARLA | **0.9.16** | UE4 계열 최신 릴리스. 0.10.x(UE5)는 8GB VRAM에서 무겁다. |
| Python | **3.10** | 0.9.16 PyPI wheel은 Windows용 3.10/3.11/3.12를 제공한다. 3.10을 쓰면 0.9.15로 내려가도 그대로 쓸 수 있다. |
| `carla` pip 패키지 | `0.9.16` | **서버와 클라이언트 버전이 정확히 같아야 한다.** 다르면 접속은 돼도 오동작한다. |

## 1. 사전 확인

- NVIDIA 드라이버: 설치되어 있음 (`nvidia-smi`로 확인)
- 디스크: CARLA 압축 파일과 압축 해제본을 합쳐 **30GB 이상** 여유 공간
- 압축 해제 경로는 짧게 잡는다. 예: `D:\CARLA_0.9.16` (Windows 경로 길이 제한 회피)

## 2. CARLA 서버 설치

1. https://github.com/carla-simulator/carla/releases/tag/0.9.16 에서 **Windows** 패키지(`CARLA_0.9.16.zip`)를 받는다.
   - Town01~05와 Town10HD는 기본 패키지에 들어 있다. AdditionalMaps는 받지 않아도 된다.
2. `D:\CARLA_0.9.16` 에 압축을 푼다. 이 폴더에 `CarlaUE4.exe`가 있어야 한다.
3. 환경변수 `CARLA_ROOT`를 등록하고 터미널을 새로 연다.
   ```powershell
   setx CARLA_ROOT "D:\CARLA_0.9.16"
   ```
4. 서버를 실행한다.
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\start_carla.ps1            # 서버 창 표시
   powershell -ExecutionPolicy Bypass -File scripts\start_carla.ps1 -OffScreen # 창 렌더링 생략 (VRAM 절약)
   ```
   스크립트가 붙이는 옵션: `-quality-level=Low -windowed -ResX=800 -ResY=600 -carla-rpc-port=2000`.
   처음 실행할 때 방화벽 허용 창이 뜨면 허용한다. 맵이 보이기까지 수십 초 걸릴 수 있다.

> 실행 직후 꺼지거나 DLL 오류가 나면 `D:\CARLA_0.9.16\Engine\Extras\Redist\en-us\UE4PrereqSetup_x64.exe`로 UE4 런타임(VC++/DirectX)을 설치한다.

## 3. Python 환경 (conda)

```powershell
cd D:\6developer\cursor\DriveLoop
conda create -n driveloop python=3.10 -y
conda activate driveloop
pip install -e ".[dev]"    # carla==0.9.16, pygame, numpy, pyyaml, pytest
pytest                     # 21개 통과하면 판단·제어 로직 정상
```

### 학습용 패키지 (2-3부터)

`pip install ultralytics`만 하면 Windows에서는 **CPU 전용 PyTorch**가 설치된다. CUDA 빌드를 먼저 설치한다.
(GPU 드라이버만 있으면 되고 CUDA Toolkit/nvcc는 필요 없다 — PyTorch가 CUDA 런타임을 포함)

```powershell
conda activate driveloop
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[train]"
python -c "import torch; print(torch.cuda.is_available())"   # True여야 함
```

base 등 다른 conda 환경의 torch/ultralytics는 공유되지 않는다. CARLA(3.10~3.12)와 같은 환경에 있어야
2-4에서 주행 루프 안에서 모델을 돌릴 수 있으므로 `driveloop` 환경에 설치한다.

## 4. 단계별 실행 확인

CARLA 서버를 띄운 상태에서 실행한다.

| 순서 | 명령 | 성공 기준 |
|---|---|---|
| 0 | `python scripts\00_check_env.py` | client와 server가 모두 `0.9.16`이고 마지막 줄에 `OK` |
| 1-1 | `python scripts\01_spawn_vehicle.py` | 서버 창에서 카메라가 차를 따라가고, 터미널에 속도가 증가하며 찍힘. 10초 뒤 자동 종료 |
| 1-2 | `python scripts\02_camera_view.py` | pygame 창에 3인칭 영상이 나오고 WASD로 운전 가능. ESC로 종료 |
| 1-3 | `python scripts\03_traffic_light_drive.py` | 차선을 따라 자율 주행. 빨간불이면 정지선 앞에서 서서히 서고, 초록불에 출발. HUD와 터미널에 상태 전이가 표시됨 |

첫 실행은 `Town01`을 로드하느라 시간이 걸린다. 1-2부터는 서버 창이 필요 없으므로 `-OffScreen`으로 띄우면 VRAM을 아낄 수 있다.

## 문제 해결

| 증상 | 조치 |
|---|---|
| `time-out ... while waiting for the simulator` | 서버가 아직 로딩 중이다. 맵이 보인 뒤 다시 실행한다. 포트 2000이 막혔는지도 확인한다. |
| 스크립트를 강제 종료한 뒤 서버가 멈춘 것처럼 보임 | 동기 모드가 남아 있는 상태다. `python scripts\00_check_env.py --reset` |
| 버전 불일치 경고 | `pip install carla==<서버 버전>` |
| VRAM 부족 또는 끊김 | `-OffScreen` 사용, `configs/sim.yaml`의 카메라 해상도 축소, 다른 GPU 프로그램 종료 |
| `pip install carla` 실패 | Python이 3.10~3.12인지 확인한다 (`python --version`). |
| HUD 글자가 깨짐 | 오버레이는 영문으로 표시한다 (pygame 기본 폰트에는 한글이 없다). |

## 데이터 수집은 Epic 품질로 (실측 2026-10-06)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_carla.ps1 -Quality Epic -OffScreen
```

| 항목 | Low | Epic |
|---|---|---|
| 비 표현 | 하늘이 검게 깨짐, 젖은 노면·반사 없음 | 흐린 하늘, 젖은 노면, 물웅덩이, 반사, 원경 안개 |
| 그림자 | 없음 | 있음 |
| 수집 속도 (1280×720 카메라 2대, NPC 15대) | 5.7 장/초 | 3.7~4.4 장/초 |
| VRAM (Town01 수집 중, 바탕화면 약 1GB 포함) | — | 3.9~4.3 GB |

- Low 품질의 비 장면은 실제 비와 다르고 하늘이 검게 깨진다. 모델이 "검은 하늘 = 비" 같은 잘못된 단서를 배울 수 있으므로 **수집에는 쓰지 않는다**.
- 서버는 기본 맵 Town10HD로 시작하며, Epic에서 이 순간 VRAM이 약 7.6GB까지 오른다. 수집기가 바로 Town01로 바꾸면 내려간다.
- 서버는 대기 중에도 GPU를 90% 넘게 쓴다. 쓰지 않을 때는 종료한다.
- 비교 이미지: `outputs/quality_compare.jpg` (위: Epic, 아래: Low)
