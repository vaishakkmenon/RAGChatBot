@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem =======================
rem Defaults (override by pre-setting URL / KEY / DATASET)
rem =======================
if not defined URL     set "URL=http://127.0.0.1:8000"
if not defined KEY     set "KEY=my-dev-key-1"
if not defined DATASET set "DATASET=%CD%\data\squad\dev-v2.0.json"

set "LIMIT=500"
set "WORKERS=4"
set "TIMEOUT=180"
set "PROGRESS=--progress --progress-interval 50"

rem =======================
rem Preflight checks
rem =======================
if not exist "scripts\squad_eval.py" (
  echo [ERROR] scripts\squad_eval.py not found.
  exit /b 1
)
if not exist "%DATASET%" (
  echo [ERROR] SQuAD file not found: %DATASET%
  exit /b 1
)

echo.
echo === Using ===
echo URL=%URL%
echo KEY=%KEY%
echo DATASET=%DATASET%
echo LIMIT=%LIMIT% WORKERS=%WORKERS% TIMEOUT=%TIMEOUT%
echo.

rem Health ping (optional)
echo Pinging /health ...
curl -s -H "X-API-Key: %KEY%" "%URL%/health" >nul 2>nul

rem =======================
rem Results folder (timestamped)
rem =======================
for /f "delims=" %%i in ('python -c "import time;print(time.strftime('%%Y%%m%%d_%%H%%M%%S'))"') do set "STAMP=%%i"
set "RESULTS=results\%STAMP%"
if not exist "%RESULTS%" mkdir "%RESULTS%"

echo Results will be saved in: %RESULTS%
echo.

rem =======================
rem Section A — Generative (grounded) abstention sweep
rem =======================
echo [A] Generative sweeps (null_threshold + reranker) ...
for %%N in (0.25 0.30 0.35 0.40) do (
  set "NT=%%N"
  set "NTS=!NT:.=p!"
  echo   - nt=!NT!, rerank=0
  python scripts\squad_eval.py --dataset "%DATASET%" --host "%URL%" --api-key "%KEY%" ^
    --grounded-only --top-k 5 --max-distance 0.60 --null-threshold !NT! ^
    --temperature 0 --workers %WORKERS% --timeout %TIMEOUT% --limit %LIMIT% %PROGRESS% ^
    --out "%RESULTS%\gen_k5_md060_nt!NTS!_rerank0.json"

  echo   - nt=!NT!, rerank=1
  python scripts\squad_eval.py --dataset "%DATASET%" --host "%URL%" --api-key "%KEY%" ^
    --grounded-only --rerank --rerank-lex-w 0.5 ^
    --top-k 5 --max-distance 0.60 --null-threshold !NT! ^
    --temperature 0 --workers %WORKERS% --timeout %TIMEOUT% --limit %LIMIT% %PROGRESS% ^
    --out "%RESULTS%\gen_k5_md060_nt!NTS!_rerank1.json"
)

rem =======================
rem Section B — Extractive baseline (your current best-ish)
rem =======================
echo.
echo [B] Extractive baseline + small sweeps ...

set "BASE_EX=--extractive --grounded-only --rerank --rerank-lex-w 0.5 --top-k 3 --max-distance 0.60 --null-threshold 0.60 --alpha 0.50 --alpha-hits 2 --support-min 0.30 --support-window 96 --span-max-distance 0.50 --temperature 0 --workers %WORKERS% --timeout %TIMEOUT% --limit %LIMIT% %PROGRESS%"

echo   - Baseline run
python scripts\squad_eval.py --dataset "%DATASET%" --host "%URL%" --api-key "%KEY%" %BASE_EX% ^
  --out "%RESULTS%\ex_k3_md060_nt060_a050_h2_sm030_sw096_sd050_rerank1.json"

rem ---- B1: support_min relax ----
for %%S in (0.30 0.25 0.20) do (
  set "SM=%%S"
  set "SMS=!SM:.=p!"
  echo   - support_min=!SM!
  python scripts\squad_eval.py --dataset "%DATASET%" --host "%URL%" --api-key "%KEY%" %BASE_EX% ^
    --support-min !SM! ^
    --out "%RESULTS%\ex_sm!SMS!.json"
)

rem ---- B2: span_max_distance relax ----
for %%D in (0.50 0.55 0.60) do (
  set "SD=%%D"
  set "SDS=!SD:.=p!"
  echo   - span_max_distance=!SD!
  python scripts\squad_eval.py --dataset "%DATASET%" --host "%URL%" --api-key "%KEY%" %BASE_EX% ^
    --span-max-distance !SD! ^
    --out "%RESULTS%\ex_sd!SDS!.json"
)

rem ---- B3: alpha / hits variants ----
for %%H in (2 1) do (
  for %%A in (0.50 0.55 0.60) do (
    set "AL=%%A"
    set "ALS=!AL:.=p!"
    set "AH=%%H"
    echo   - alpha=!AL!, hits=!AH!
    python scripts\squad_eval.py --dataset "%DATASET%" --host "%URL%" --api-key "%KEY%" %BASE_EX% ^
      --alpha !AL! --alpha-hits !AH! ^
      --out "%RESULTS%\ex_a!ALS!_h!AH!.json"
  )
)

rem ---- B4: top_k 3 vs 4 ----
for %%K in (3 4) do (
  echo   - top_k=%%K
  python scripts\squad_eval.py --dataset "%DATASET%" --host "%URL%" --api-key "%KEY%" %BASE_EX% ^
    --top-k %%K ^
    --out "%RESULTS%\ex_k%%K.json"
)

rem =======================
rem Section C — Summarize best configs
rem =======================
echo.
echo [C] Building leaderboard ...

rem Write pick_best.py line-by-line to avoid () grouping issues
> "%RESULTS%\pick_best.py" echo import json,glob,os
>> "%RESULTS%\pick_best.py" echo folder = r"%RESULTS%"
>> "%RESULTS%\pick_best.py" echo files = sorted(glob.glob(os.path.join(folder, "*.json")))
>> "%RESULTS%\pick_best.py" echo rows=[]
>> "%RESULTS%\pick_best.py" echo for f in files:
>> "%RESULTS%\pick_best.py" echo\ \ \ \ try:
>> "%RESULTS%\pick_best.py" echo\ \ \ \ \ \ \ \ d=json.load(open(f,"r",encoding="utf-8"))
>> "%RESULTS%\pick_best.py" echo\ \ \ \ except Exception:
>> "%RESULTS%\pick_best.py" echo\ \ \ \ \ \ \ \ continue
>> "%RESULTS%\pick_best.py" echo\ \ \ \ cfg=d.get("settings",{})
>> "%RESULTS%\pick_best.py" echo\ \ \ \ met=d.get("overall",{})
>> "%RESULTS%\pick_best.py" echo\ \ \ \ ans=d.get("answerable",{})
>> "%RESULTS%\pick_best.py" echo\ \ \ \ un=d.get("unanswerable",{})
>> "%RESULTS%\pick_best.py" echo\ \ \ \ lat=d.get("latency",{})
>> "%RESULTS%\pick_best.py" echo\ \ \ \ rows.append((
>> "%RESULTS%\pick_best.py" echo\ \ \ \ \ \ \ \ os.path.basename(f),
>> "%RESULTS%\pick_best.py" echo\ \ \ \ \ \ \ \ float(met.get("F1",0.0)),
>> "%RESULTS%\pick_best.py" echo\ \ \ \ \ \ \ \ float(met.get("EM",0.0)),
>> "%RESULTS%\pick_best.py" echo\ \ \ \ \ \ \ \ float(ans.get("F1") or 0.0),
>> "%RESULTS%\pick_best.py" echo\ \ \ \ \ \ \ \ float(un.get("NoAns_Accuracy") or 0.0),
>> "%RESULTS%\pick_best.py" echo\ \ \ \ \ \ \ \ lat.get("p50_ms"),
>> "%RESULTS%\pick_best.py" echo\ \ \ \ \ \ \ \ bool(cfg.get("extractive")),
>> "%RESULTS%\pick_best.py" echo\ \ \ \ \ \ \ \ bool(cfg.get("rerank")),
>> "%RESULTS%\pick_best.py" echo\ \ \ \ \ \ \ \ cfg.get("top_k"),
>> "%RESULTS%\pick_best.py" echo\ \ \ \ \ \ \ \ cfg.get("max_distance"),
>> "%RESULTS%\pick_best.py" echo\ \ \ \ \ \ \ \ cfg.get("null_threshold"),
>> "%RESULTS%\pick_best.py" echo\ \ \ \ \ \ \ \ cfg.get("alpha"),
>> "%RESULTS%\pick_best.py" echo\ \ \ \ \ \ \ \ cfg.get("alpha_hits"),
>> "%RESULTS%\pick_best.py" echo\ \ \ \ \ \ \ \ cfg.get("support_min"),
>> "%RESULTS%\pick_best.py" echo\ \ \ \ \ \ \ \ cfg.get("support_window"),
>> "%RESULTS%\pick_best.py" echo\ \ \ \ \ \ \ \ cfg.get("span_max_distance"),
>> "%RESULTS%\pick_best.py" echo\ \ \ \ ))
>> "%RESULTS%\pick_best.py" echo.
>> "%RESULTS%\pick_best.py" echo def show(title, keyidx, reverse=True, limit=10):
>> "%RESULTS%\pick_best.py" echo\ \ \ \ print("\n=== " + title + " ===")
>> "%RESULTS%\pick_best.py" echo\ \ \ \ for r in sorted(rows, key=lambda x:x[keyidx], reverse=reverse)[:limit]:
>> "%RESULTS%\pick_best.py" echo\ \ \ \ \ \ \ \ print(f"{r[0]:45s}  F1={r[1]:.3f}  EM={r[2]:.3f}  AnsF1={r[3]:.3f}  NoAns={r[4]:.3f}  p50={r[5]}  EX={int(r[6])}  RR={int(r[7])}  k={r[8]}  md={r[9]}  nt={r[10]}  a={r[11]}  h={r[12]}  sm={r[13]}  sw={r[14]}  sd={r[15]}")
>> "%RESULTS%\pick_best.py" echo if rows:
>> "%RESULTS%\pick_best.py" echo\ \ \ \ show("Top 10 by Overall F1", 1, True)
>> "%RESULTS%\pick_best.py" echo\ \ \ \ show("Top 10 by NoAns Accuracy", 4, True)

python "%RESULTS%\pick_best.py"

echo.
echo Done. Best runs are under: %RESULTS%
endlocal