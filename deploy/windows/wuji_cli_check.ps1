# Installs Wuji CLI (official installer) if missing, then runs its diagnostics for the Wuji Hand 2
# and writes everything to runs\tactile\wuji_cli_<time>.txt. No firmware is changed.
$ErrorActionPreference = "Continue"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$outDir = Join-Path $repo "runs\tactile"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$out = Join-Path $outDir ("wuji_cli_{0:yyyy-MM-dd_HHmmss}.txt" -f (Get-Date))
function Log($s) { Write-Host $s; Add-Content -Path $out -Value $s }
Log "# wuji-cli check $(Get-Date -Format s)"

function Find-Wuji {
    $c = Get-Command wuji -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    foreach ($p in @("$env:LOCALAPPDATA\Programs\wuji\wuji.exe", "$env:LOCALAPPDATA\wuji\bin\wuji.exe", "$env:USERPROFILE\.wuji\bin\wuji.exe",
                     "$env:LOCALAPPDATA\Programs\wuji-cli\wuji.exe", "$env:ProgramFiles\wuji\wuji.exe", "$env:USERPROFILE\.local\bin\wuji.exe")) {
        if (Test-Path $p) { return $p }
    }
    $found = Get-ChildItem -Path $env:LOCALAPPDATA, $env:USERPROFILE -Filter wuji.exe -Recurse -Depth 4 -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($found) { return $found.FullName }
    return $null
}

$wuji = Find-Wuji
if (-not $wuji) {
    Log "[install] Wuji CLI not found; running the official installer: irm https://get.wuji.tech/cli.ps1 | iex"
    try { Invoke-Expression (Invoke-RestMethod https://get.wuji.tech/cli.ps1) 2>&1 | ForEach-Object { Log "[install] $_" } }
    catch { Log "[install] installer error: $_" }
    # refresh PATH for this process
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
    $wuji = Find-Wuji
}
if (-not $wuji) { Log "[install] wuji.exe still not found after install"; Log "# done: $out"; exit 1 }
Log "[cli] using $wuji"

function Run($argsList) {
    Log ""
    Log ("$ wuji " + ($argsList -join " "))
    try {
        $o = & $wuji @argsList 2>&1 | Out-String
        $o -split "`r?`n" | ForEach-Object { Log $_ }
    } catch { Log "ERROR: $_" }
}
Run @("--version")
Run @("devices")
Run @("ping")
Run @("doctor")
Run @("upgrade", "--check")
Run @("get", "node_online")
Run @("get", "firmware_version")
Run @("get", "hw_version")
Run @("get", "health_status")
Run @("get", "slave_version")
foreach ($f in @("thumb", "index", "middle", "ring", "pinky")) {
    Run @("get", "$f.tactile.tactile_type")
    Run @("get", "$f.tactile.tactile_state")
}
Run @("logs", "--help")
Log "# done: $out"
