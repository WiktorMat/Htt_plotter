# Instructions for micromamba installation on Windows
cd $HOME
mkdir micromamba
cd micromamba
Invoke-WebRequest -Uri https://micro.mamba.pm/api/micromamba/win-64/latest -OutFile micromamba.tar.bz2
tar -xvjf micromamba.tar.bz2
$env:MAMBA_ROOT_PREFIX = "$HOME\micromamba"

[System.Environment]::SetEnvironmentVariable(
    "MAMBA_ROOT_PREFIX",
    "$HOME\micromamba",
    "User"
)

.\Library\bin\micromamba.exe shell init -s powershell
. $PROFILE