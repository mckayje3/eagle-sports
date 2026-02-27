# PowerShell script to set up Windows Task Scheduler for daily updates
# Run as Administrator: Right-click PowerShell -> Run as Administrator
# Then run: .\setup_daily_task.ps1

$taskName = "SportsDataDailyUpdate"
$taskPath = "C:\Users\jbeast\documents\coding\sports\daily_update.bat"
$logPath = "C:\Users\jbeast\documents\coding\sports\logs"

# Create logs directory if it doesn't exist
if (!(Test-Path $logPath)) {
    New-Item -ItemType Directory -Path $logPath -Force
}

# Remove existing task if it exists
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue

# Create the action (run the batch file)
$action = New-ScheduledTaskAction -Execute $taskPath -WorkingDirectory "C:\Users\jbeast\documents\coding\sports"

# Create trigger - Daily at 9:00 AM
$trigger = New-ScheduledTaskTrigger -Daily -At 9:00AM

# Create settings
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -RunOnlyIfNetworkAvailable

# Register the task
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Description "Daily update of sports odds, predictions, and cloud sync"

Write-Host ""
Write-Host "============================================================"
Write-Host "Task '$taskName' created successfully!"
Write-Host "============================================================"
Write-Host ""
Write-Host "The task will run daily at 9:00 AM"
Write-Host ""
Write-Host "To test it now, run:"
Write-Host "  Start-ScheduledTask -TaskName '$taskName'"
Write-Host ""
Write-Host "To view the task:"
Write-Host "  Get-ScheduledTask -TaskName '$taskName' | Format-List"
Write-Host ""
Write-Host "To remove the task:"
Write-Host "  Unregister-ScheduledTask -TaskName '$taskName'"
Write-Host ""
