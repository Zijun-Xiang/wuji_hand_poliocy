# Upgrades the Wuji Hand 2 firmware with Wuji CLI (wuji upgrade), logging to runs\tactile\wuji_upgrade_<time>.txt,
# then re-reads firmware_version / node_online and re-runs deploy\tactile_probe.py.
# Keep the hand powered and on Ethernet for the whole run. Answer any [Y/n] prompt in this window.
$ErrorActionPreference = "Continue"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$outDir = Join-Path $repo "runs\tactile"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$out = Join-Path $outDir ("wuji_upgrade_{0:yyyy-MM-dd_HHmmss}.txt" -f (Get-Date))
function Log($s) { Write-Host $s; Add-Content -Path $out -Value $s }
$wuji = (Get-Command wuji -ErrorAction SilentlyContinue).Source
if (-not $wuji -and (Test-Path "$env:LOCALAPPDATA\Programs\Wuji\bin\wuji.exe")) { $wuji = "$env:LOCALAPPDATA\Programs\Wuji\bin\wuji.exe" }
if (-not $wuji) { Log "wuji.exe not found - run wuji_cli_check.bat first"; exit 1 }
Log "# wuji upgrade $(Get-Date -Format s)  using $wuji"
function Run($argsList) {
    Log ""; Log ("$ wuji " + ($argsList -join " "))
    & $wuji @argsList 2>&1 | ForEach-Object { $line = "$_"; Write-Host $line; Add-Content -Path $out -Value $line }
    Log ("[exit code " + $LASTEXITCODE + "]")
}
Run @("get", "firmware_version")
$help = (& $wuji upgrade --help 2>&1 | Out-String)
Log "$ wuji upgrade --help"; Log $help
$upArgs = @("upgrade")
if ($help -match "--sn") { $upArgs += @("--sn", "WH2KA01260824017") }
if ($help -match "--yes") { $upArgs += "--yes" } elseif ($help -match "(?m)^\s*-y[,\s]") { $upArgs += "-y" }
Log "[upgrade] starting: wuji $($upArgs -join ' ')   (do not power off the hand)"
Run $upArgs
Log "[upgrade] waiting 25 s for the hand to restart"
Start-Sleep -Seconds 25
Run @("devices")
Run @("get", "firmware_version")
Run @("get", "node_online")
Run @("upgrade", "--check")
$py = Join-Path $repo ".venv-deploy\Scripts\python.exe"
if (Test-Path $py) {
    Log ""; Log "$ tactile_probe.py"
    & $py (Join-Path $repo "deploy\tactile_probe.py") 2>&1 | ForEach-Object { $line = "$_"; Write-Host $line; Add-Content -Path $out -Value $line }
}
Log "# done: $out"
