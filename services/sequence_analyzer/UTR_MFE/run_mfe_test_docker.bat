@echo off
REM Quick script to build and run MFE test in Docker environment (Windows)

echo ==========================================
echo 5' UTR MFE Test - Docker Environment
echo ==========================================
echo.

REM Check if Docker is available
docker --version >nul 2>&1
if errorlevel 1 (
    echo Error: Docker is not installed or not in PATH
    exit /b 1
)

echo Docker is available
echo.

REM Build the image
echo Building Docker image with ViennaRNA...
echo (This may take 5-10 minutes on first build)
echo.

docker build -t sequence-analyzer -f services/sequence_analyzer/Dockerfile .
if errorlevel 1 (
    echo Failed to build Docker image
    exit /b 1
)

echo.
echo Docker image built successfully
echo.

REM Verify ViennaRNA
echo Verifying ViennaRNA installation...
docker run --rm sequence-analyzer RNAfold --version
echo.

REM Run the test
echo ==========================================
echo Running MFE Test Demo...
echo ==========================================
echo.

docker run --rm -v "%cd%:/workspace" -w /workspace sequence-analyzer python3 test_utr_mfe_demo.py

echo.
echo ==========================================
echo Test completed successfully!
echo ==========================================
echo.
echo To run interactive tests, use:
echo   docker run --rm -it -v "%cd%:/workspace" -w /workspace sequence-analyzer bash
echo.

pause

