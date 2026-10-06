# Helper suspend/resume pour les runs LBM sweep_energy_matched
# Usage :
#   powershell -ExecutionPolicy Bypass -File suspend_resume_lbm.ps1 -ProcessId 22988 -Action Suspend
#   powershell -ExecutionPolicy Bypass -File suspend_resume_lbm.ps1 -ProcessId 22988 -Action Resume
#
# Contexte : le 20/09 21:55, le run N512 (PID 22988) a ete suspendu pour accelérer
# le jumeau N256 (PID 5408). Etat conserve en RAM ; reprendre apres fin du N256.

param(
  [Parameter(Mandatory=$true)][int]$ProcessId,
  [Parameter(Mandatory=$true)][ValidateSet("Suspend","Resume")][string]$Action
)

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class ProcSus {
  [DllImport("ntdll.dll", SetLastError=true)]
  public static extern int NtSuspendProcess(IntPtr h);
  [DllImport("ntdll.dll", SetLastError=true)]
  public static extern int NtResumeProcess(IntPtr h);
}
"@

$proc = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
if (-not $proc) { Write-Error "Process $ProcessId introuvable"; exit 1 }
$h = $proc.Handle
$ret = 0
if ($Action -eq "Suspend") { $ret = [ProcSus]::NtSuspendProcess($h) } else { $ret = [ProcSus]::NtResumeProcess($h) }
"$Action PID $ProcessId -> retour $ret (0=OK)"

# Verification : apres un Resume, le CPU doit re-augmenter
if ($Action -eq "Resume") {
  $c1 = (Get-Process -Id $ProcessId).CPU
  Start-Sleep -Seconds 10
  $c2 = (Get-Process -Id $ProcessId).CPU
  "CPU delta 10s apres Resume = " + [math]::Round($c2 - $c1, 1) + " s (attendu > 0)"
}