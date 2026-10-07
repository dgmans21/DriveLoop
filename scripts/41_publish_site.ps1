# Publish docs/ (code + build output from 40_build_site.py) to the gh-pages branch as ONE fresh commit.
# Old videos never pile up in history: gh-pages is replaced every time. main is not touched.
#
#   python scripts/40_build_site.py
#   powershell -ExecutionPolicy Bypass -File scripts/41_publish_site.ps1
#   git push -f origin gh-pages          # GitHub: Settings > Pages > Branch gh-pages, folder /
param(
    [string]$Branch = "gh-pages",
    [string]$Message = ""
)
$ErrorActionPreference = "Stop"
$root = (git rev-parse --show-toplevel).Trim()
$docs = Join-Path $root "docs"
if (-not (Test-Path (Join-Path $docs "assets/drive/index.json"))) {
    throw "docs/assets is not built. Run: python scripts/40_build_site.py"
}

$src = (git -C $root rev-parse --short HEAD).Trim()
$dirty = if (git -C $root status --porcelain -- docs scripts/40_build_site.py) { " (+uncommitted changes)" } else { "" }
if (-not $Message) { $Message = "Publish site from main@$src$dirty" }

$wt = Join-Path ([System.IO.Path]::GetTempPath()) "driveloop-gh-pages"
if (Test-Path $wt) { git -C $root worktree remove --force $wt 2>$null; Remove-Item -Recurse -Force $wt -ErrorAction SilentlyContinue }
git -C $root worktree prune

try {
    # orphan branch in a temporary worktree: no parent commit -> no history to accumulate
    git -C $root worktree add --detach $wt | Out-Null
    git -C $wt checkout --orphan "$Branch-tmp" | Out-Null
    git -C $wt rm -rf --quiet . 2>$null
    Get-ChildItem -Force $wt | Where-Object { $_.Name -ne ".git" } | Remove-Item -Recurse -Force

    Copy-Item -Recurse -Force (Join-Path $docs "*") $wt
    New-Item -ItemType File -Force (Join-Path $wt ".nojekyll") | Out-Null

    git -C $wt add -A
    git -C $wt commit --quiet -m $Message -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
    if (git -C $root branch --list $Branch) { git -C $root branch -D $Branch | Out-Null }
    git -C $wt branch -m $Branch
    $sha = (git -C $wt rev-parse --short HEAD).Trim()
}
finally {
    git -C $root worktree remove --force $wt 2>$null
    git -C $root worktree prune
}

$size = (Get-ChildItem -Recurse -File $docs | Measure-Object Length -Sum).Sum / 1MB
Write-Host ("[publish] {0} = 1 commit {1}  ({2:N1} MB)  <- main@{3}{4}" -f $Branch, $sha, $size, $src, $dirty)
Write-Host "[publish] next: git push -f origin $Branch"
