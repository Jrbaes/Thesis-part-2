Set-Location $PSScriptRoot
while ($true) {
    Write-Host "[launcher] Starting Streamlit app..." -ForegroundColor Cyan
    python -m streamlit run app.py
    $code = $LASTEXITCODE
    if ($code -eq 0) {
        Write-Host "[launcher] App exited cleanly (exit 0). Stopping." -ForegroundColor Green
        break
    }
    Write-Host "[launcher] App crashed (exit $code). Restarting in 2 seconds..." -ForegroundColor Yellow
    Start-Sleep -Seconds 1
}
