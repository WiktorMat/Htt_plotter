# Htt_plotter: jak pobrać gałąź i zainstalować jako pakiet

## 1) Jak zaciągnąć tylko jedną gałąź z GitHuba

### Opcja A — nowe klonowanie (tylko jedna gałąź)

```bash
git clone --single-branch --branch <NAZWA_GALEZI> https://github.com/WiktorMat/Htt_plotter.git
```

Przykład:

```bash
git clone --single-branch --branch plotter/streaming-logs-cleanup https://github.com/WiktorMat/Htt_plotter.git
```

### Opcja B — masz już repo lokalnie i chcesz tylko dociągnąć nową gałąź

W katalogu repo:

```bash
git fetch origin <NAZWA_GALEZI>
git switch -c <NAZWA_GALEZI> --track origin/<NAZWA_GALEZI>
```

Przykład:

```bash
git fetch origin plotter/streaming-logs-cleanup
git switch -c plotter/streaming-logs-cleanup --track origin/plotter/streaming-logs-cleanup
```

## 2) Instalacja kodu jako pakiet (editable install)

W katalogu `Htt_plotter/`:

```bash
python3 -m pip install -U pip
python3 -m pip install -e .
```

Szybka weryfikacja importu:

```bash
python3 -c "import htt_plotter; print(htt_plotter.__file__)"
```

Uruchomienie przykładu:

```bash
python3 source/test_run.py
```

## 3) Micromamba (opcjonalnie)

### Linux (np. lxplus)

Przykładowe środowisko (conda-forge):

```bash
micromamba create -n htt_plotter -c conda-forge \
  python=3.11 numpy pandas pyarrow pyyaml matplotlib
micromamba activate htt_plotter
```

Potem instalacja pakietu jak wyżej: `python3 -m pip install -e .`.

### Windows (PowerShell) — instalacja micromamba

```powershell
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
```
