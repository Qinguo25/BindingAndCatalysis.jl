$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$threads = if ($env:BNC_THREADS) { $env:BNC_THREADS } else { "5" }
$cdn = if ($env:BNC_CDN_N) { $env:BNC_CDN_N } else { "5" }
$solver = if ($env:BNC_CDN_SOLVER) { $env:BNC_CDN_SOLVER } else { "dag" }
$heartbeatSeconds = if ($env:BNC_HEARTBEAT_SECONDS) { $env:BNC_HEARTBEAT_SECONDS } else { "60" }
$timeoutSeconds = if ($env:BNC_TIMEOUT_SECONDS) { [int]$env:BNC_TIMEOUT_SECONDS } else { 23400 }

$statusPath = Join-Path $repoRoot "test\cdn${cdn}_${solver}_status.json"
$resultPath = Join-Path $repoRoot "test\cdn${cdn}_${solver}_result.json"
$stdoutPath = Join-Path $repoRoot "test\cdn${cdn}_${solver}_stdout.log"
$stderrPath = Join-Path $repoRoot "test\cdn${cdn}_${solver}_stderr.log"
$launcherPath = Join-Path $repoRoot "test\cdn${cdn}_${solver}_launcher.log"

function Stop-ProcessTree {
    param(
        [Parameter(Mandatory = $true)]
        [int]$RootPid
    )

    $all = Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId
    $childrenByParent = @{}
    foreach ($proc in $all) {
        if (-not $childrenByParent.ContainsKey($proc.ParentProcessId)) {
            $childrenByParent[$proc.ParentProcessId] = New-Object System.Collections.Generic.List[int]
        }
        $childrenByParent[$proc.ParentProcessId].Add([int]$proc.ProcessId)
    }

    $toVisit = New-Object System.Collections.Generic.Stack[int]
    $toVisit.Push($RootPid)
    $seen = New-Object System.Collections.Generic.HashSet[int]
    $ordered = New-Object System.Collections.Generic.List[int]

    while ($toVisit.Count -gt 0) {
        $pid = $toVisit.Pop()
        if (-not $seen.Add($pid)) {
            continue
        }
        $ordered.Add($pid)
        if ($childrenByParent.ContainsKey($pid)) {
            foreach ($child in $childrenByParent[$pid]) {
                $toVisit.Push($child)
            }
        }
    }

    [array]::Reverse($ordered.ToArray()) | Out-Null
    foreach ($pid in ($ordered | Sort-Object -Descending)) {
        try {
            Stop-Process -Id $pid -Force -ErrorAction Stop
        } catch {
        }
    }
}

$startLine = "$(Get-Date -Format s) starting overnight run: cdn=$cdn solver=$solver threads=$threads timeout_seconds=$timeoutSeconds"
Set-Content -Path $launcherPath -Value $startLine

$command = @"
`$env:JULIA_NUM_THREADS='$threads'
`$env:BNC_CDN_N='$cdn'
`$env:BNC_CDN_SOLVER='$solver'
`$env:BNC_HEARTBEAT_SECONDS='$heartbeatSeconds'
`$env:BNC_STATUS_PATH='$statusPath'
`$env:BNC_RESULT_PATH='$resultPath'
julia --project=. test/SISO_test/long_runs/cdn_overnight_benchmark.jl
"@

$process = Start-Process -FilePath "pwsh" -ArgumentList "-NoLogo", "-NoProfile", "-Command", $command -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath

if ($process.WaitForExit($timeoutSeconds * 1000)) {
    $endLine = "$(Get-Date -Format s) completed with exit_code=$($process.ExitCode)"
    Add-Content -Path $launcherPath -Value $endLine
    exit $process.ExitCode
}

try {
    Stop-ProcessTree -RootPid $process.Id
} catch {
}

$timeoutInfo = @{
    finished_at = (Get-Date -Format s)
    stage = "timeout"
    timeout_seconds = $timeoutSeconds
    cdn = [int]$cdn
    condition_solver = $solver
    julia_threads = [int]$threads
    status_path = $statusPath
    result_path = $resultPath
    stdout_path = $stdoutPath
    stderr_path = $stderrPath
}

$timeoutJson = $timeoutInfo | ConvertTo-Json -Depth 4
Set-Content -Path $resultPath -Value $timeoutJson
Set-Content -Path $statusPath -Value $timeoutJson
Add-Content -Path $launcherPath -Value "$(Get-Date -Format s) timed out and process was terminated"

exit 124
