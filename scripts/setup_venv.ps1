# PowerShell Setup Script for Agentic AI Project
# Run this script from the project root directory:
#   cd C:\Users\T10007\Desktop\Ltm_projects\Agentic_AI
#   .\scripts\setup_venv.ps1

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Agentic AI — Virtual Environment Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Step 1: Check Python
Write-Host "`n[1/6] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Python not found. Install Python 3.9+ and add to PATH." -ForegroundColor Red
    exit 1
}

# Step 2: Create virtual environment
$venvName = "venvagentic"
if (Test-Path $venvName) {
    Write-Host "`n[2/6] Virtual environment '$venvName' already exists." -ForegroundColor Green
    Write-Host "  Reusing existing environment." -ForegroundColor Green
} else {
    Write-Host "`n[2/6] Creating virtual environment: $venvName" -ForegroundColor Yellow
    python -m venv $venvName
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERROR: Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
    Write-Host "  Created: $venvName" -ForegroundColor Green
}

# Step 3: Activate
Write-Host "`n[3/6] Activating virtual environment..." -ForegroundColor Yellow
$activateScript = ".\$venvName\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
    Write-Host "  Activated: $venvName" -ForegroundColor Green
} else {
    Write-Host "  ERROR: Activation script not found: $activateScript" -ForegroundColor Red
    exit 1
}

# Step 4: Upgrade pip
Write-Host "`n[4/6] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip
Write-Host "  pip upgraded." -ForegroundColor Green

# Step 5: Install dependencies
Write-Host "`n[5/6] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "  WARNING: Some packages may have failed. Check output above." -ForegroundColor Yellow
} else {
    Write-Host "  Dependencies installed successfully." -ForegroundColor Green
}

# Step 6: Install the package in editable mode
Write-Host "`n[6/6] Installing industrial_health package (editable)..." -ForegroundColor Yellow
pip install -e .
Write-Host "  Package installed." -ForegroundColor Green

# Final verification
Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "  SETUP COMPLETE" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Python version:" -ForegroundColor White
python --version
Write-Host ""
Write-Host "pip version:" -ForegroundColor White
pip --version
Write-Host ""
Write-Host "To activate later, run:" -ForegroundColor Yellow
Write-Host "  .\venvagentic\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "Quick test (run after activation):" -ForegroundColor Yellow
Write-Host "  python -c `"from industrial_health.data.loader import get_test_info; print(get_test_info(1))`"" -ForegroundColor White
Write-Host ""
Write-Host "Run tests:" -ForegroundColor Yellow
Write-Host "  pytest tests/ -v" -ForegroundColor White
