# A/B 对比：章节级去重 开 vs 关
# 用法（项目根目录）：pwsh -File eval\ab_dedupe.ps1

Set-Location 'C:\Users\27173\Desktop\Personal knowledge'
$py = '.\.venv\Scripts\python.exe'

function Invoke-Eval {
    param([string[]]$Sets, [string]$Mode, [string]$Dedupe)
    $env:RAG_SECTION_DEDUPE = $Dedupe
    $argv = @('-B', 'eval\run_eval.py', '--mode', $Mode)
    foreach ($s in $Sets) { $argv += '--set'; $argv += $s }
    $out = & $py @argv 2>&1
    $line = ($out | Select-String '共 \d+ 题' | Select-Object -Last 1)
    if ($line) { return $line.ToString().Trim() } else { return '(无输出)' }
}

$cases = @(
    @{ label = '基础 17 题  hybrid'; sets = @('cs5', 'drawing12'); mode = 'hybrid' },
    @{ label = '基础 17 题  dense '; sets = @('cs5', 'drawing12'); mode = 'dense' },
    @{ label = '难题  8 题  hybrid'; sets = @('hard8'); mode = 'hybrid' },
    @{ label = '难题  8 题  dense '; sets = @('hard8'); mode = 'dense' }
)

Write-Host ''
Write-Host ('=' * 96)
Write-Host '去重【关】(改动前行为)'
Write-Host ('=' * 96)
$before = @{}
foreach ($c in $cases) {
    $r = Invoke-Eval -Sets $c.sets -Mode $c.mode -Dedupe '0'
    $before[$c.label] = $r
    Write-Host ("{0}  {1}" -f $c.label, $r)
}

Write-Host ''
Write-Host ('=' * 96)
Write-Host '去重【开】(改动后)'
Write-Host ('=' * 96)
foreach ($c in $cases) {
    $r = Invoke-Eval -Sets $c.sets -Mode $c.mode -Dedupe '1'
    Write-Host ("{0}  {1}" -f $c.label, $r)
}

Remove-Item Env:\RAG_SECTION_DEDUPE -ErrorAction SilentlyContinue
Write-Host ''
