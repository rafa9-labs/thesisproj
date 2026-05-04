@echo off
setlocal enabledelayedexpansion

echo =============================================
echo  KodaQuant - Self-Signed Code Signing Setup
echo =============================================
echo.
echo This script creates a self-signed code signing certificate
echo for development builds. For production, use a proper
echo code signing certificate from a CA (DigiCert, Sectigo, etc.).
echo.

REM Check for Windows SDK signtool
where signtool >nul 2>&1
if errorlevel 1 (
    echo [WARN] signtool.exe not found on PATH.
    echo       Install Windows 10 SDK from:
    echo       https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/
    echo.
    echo       Alternatively, install via Visual Studio Installer:
    echo       "Windows 10 SDK" workload.
    echo.
    echo [INFO] Skipping code signing. The build will work without signing,
    echo       but Windows will show "Unknown Publisher" on the installer.
    echo.
    echo       To enable signing later:
    echo       1. Install Windows 10 SDK
    echo       2. Run this script again to create a self-signed cert
    echo       3. Set CSC_LINK and CSC_KEY_PASSWORD env vars
    echo       4. Change signAndEditExecutable to true in electron-builder.yml
    exit /b 0
)

echo [INFO] signtool found. Creating self-signed certificate...

REM Create self-signed certificate (valid for 1 year, for code signing)
set CERT_NAME="KodaQuant Dev Code Signing"
set CERT_FILE=%~dp0..\build\kodaquant-dev.p12
set CERT_PASS=kodaquant-dev-2026

REM Check if cert already exists in Personal store
certutil -store My "KodaQuant Dev Code Signing" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Certificate already exists in Personal store. Skipping creation.
) else (
    echo [INFO] Creating self-signed code signing certificate...
    REM Create self-signed cert with makecert alternative using PowerShell
    powershell -Command "$cert = New-SelfSignedCertificate -Type CodeSigningCert -Subject 'CN=KodaQuant Dev Code Signing' -CertStoreLocation 'Cert:\CurrentUser\My' -HashAlgorithm SHA256 -KeyAlgorithm RSA -KeyLength 2048 -NotAfter (Get-Date).AddYears(1); $pwd = ConvertTo-SecureString -String '%CERT_PASS%' -Force -AsPlainText; Export-PfxCertificate -Cert $cert -FilePath '%CERT_FILE%' -Password $pwd | Out-Null; Write-Output 'Certificate exported to: %CERT_FILE%'"

    if errorlevel 1 (
        echo [ERROR] Failed to create self-signed certificate.
        exit /b 1
    )

    REM Add cert to Trusted Root (requires admin)
    echo [INFO] Adding certificate to Trusted Root CA store...
    echo [WARN] This requires Administrator privileges.
    powershell -Command "Import-Certificate -FilePath '%CERT_FILE%' -CertStoreLocation 'Cert:\CurrentUser\Root' | Out-Null; Write-Output 'Added to Trusted Root.'"

    if errorlevel 1 (
        echo [WARN] Could not add to Trusted Root. Run as Administrator for local trust.
    )
)

echo.
echo =============================================
echo  Code signing setup complete!
echo =============================================
echo.
echo  To sign production builds:
echo    set CSC_LINK=%CERT_FILE%
echo    set CSC_KEY_PASSWORD=%CERT_PASS%
echo    echo signAndEditExecutable: true in electron-builder.yml
echo.
echo  Then rebuild:
echo    scripts\build_electron.bat