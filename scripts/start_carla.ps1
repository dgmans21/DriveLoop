# CARLA 서버 실행
#   powershell -ExecutionPolicy Bypass -File scripts\start_carla.ps1                          # 저품질 + 서버 창 (1-1 확인용)
#   powershell -ExecutionPolicy Bypass -File scripts\start_carla.ps1 -OffScreen               # 서버 창 렌더링 생략 (VRAM 절약)
#   powershell -ExecutionPolicy Bypass -File scripts\start_carla.ps1 -Quality Epic -OffScreen # 데이터 수집용 고품질
param(
    [string]$CarlaRoot = $env:CARLA_ROOT,
    [int]$Port = 2000,
    [ValidateSet("Low", "Epic")]
    [string]$Quality = "Low",
    [switch]$OffScreen
)

if (-not $CarlaRoot) { throw "CARLA_ROOT 환경변수 또는 -CarlaRoot 인자로 CARLA 설치 폴더를 지정하세요." }
$exe = Join-Path $CarlaRoot "CarlaUE4.exe"
if (-not (Test-Path $exe)) { throw "CarlaUE4.exe 를 찾을 수 없음: $exe" }

$carlaArgs = @("-quality-level=$Quality", "-carla-rpc-port=$Port", "-windowed", "-ResX=800", "-ResY=600")
if ($OffScreen) { $carlaArgs += "-RenderOffScreen" }

Write-Host "Starting: $exe $($carlaArgs -join ' ')"
Start-Process -FilePath $exe -ArgumentList $carlaArgs
