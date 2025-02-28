@echo off
:: GitHub Repository Backup Script
:: Created for Moonscrape27_2025 repository
:: Backup location: C:\GitHub_Backups

:: Set variables
set REPO_URL=https://github.com/TheManiacalCoder/Moonscrape27_2025.git
set BACKUP_DIR=C:\GitHub_Backups\Moonscrape27_2025
set TIMESTAMP=%date:/=-%_%time::=-%
set TIMESTAMP=%TIMESTAMP: =0%
set BACKUP_NAME=Moonscrape27_2025_%TIMESTAMP%

:: Create backup directory if it doesn't exist
if not exist "%BACKUP_DIR%" (
    echo Creating backup directory...
    mkdir "%BACKUP_DIR%"
)

:: Navigate to backup directory
cd /d "%BACKUP_DIR%"

:: Clone or pull the repository
if exist "%BACKUP_DIR%\.git" (
    echo Updating existing repository...
    git pull origin main
) else (
    echo Cloning repository for the first time...
    git clone %REPO_URL% .
)

:: Create zip backup
echo Creating backup archive...
powershell -Command "Compress-Archive -Path '%BACKUP_DIR%' -DestinationPath '%BACKUP_DIR%\%BACKUP_NAME%.zip'"

:: Clean up old backups (keep last 5)
echo Cleaning up old backups...
for /f "skip=5 delims=" %%f in ('dir /b /o-d /tw "%BACKUP_DIR%\*.zip"') do (
    echo Deleting old backup: %%f
    del "%BACKUP_DIR%\%%f"
)

echo Backup completed successfully!
echo Backup saved as: %BACKUP_DIR%\%BACKUP_NAME%.zip
pause 