"""Run test files individually with per-file timeout and proper encoding."""
import subprocess, os, time

VENV = r"C:\Users\Chris\Documents\Code\bluesky-PettingZoo\.venv\Scripts\python.exe"
ROOT  = r"C:\Users\Chris\Documents\Code\bluesky-PettingZoo"
SKIP  = {"test_ablation.py","test_algorithm_config.py","test_algorithm_factory.py",
         "test_train_cli.py","test_train_ppo_scenarios.py","test_train_render.py",
         "test_train_script.py","test_wrapper_switch.py","test_baseline_evaluation.py",
         "test_run_baselines.py","test_evaluate_all.py","test_evaluate_script.py"}
TIMEOUT = 90
env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}

files = sorted(f for f in os.listdir(os.path.join(ROOT,"tests"))
               if f.startswith("test_") and f.endswith(".py") and f not in SKIP)
passed=failed=timeout_cnt=err_cnt=0; fail_list=[]; timeout_list=[]; err_list=[]

for i, f in enumerate(files, 1):
    print(f"[{i}/{len(files)}] {f} ... ", end="", flush=True)
    t0 = time.time()
    try:
        r = subprocess.run([VENV,"-m","pytest",os.path.join("tests",f),"--no-cov","-q","--tb=line"],
                           cwd=ROOT, timeout=TIMEOUT, env=env,
                           capture_output=True)
        elapsed = time.time()-t0
        out = (r.stdout or b"").decode("utf-8",errors="replace")
        err = (r.stderr or b"").decode("utf-8",errors="replace")
        all_out = out + "\n" + err
        summary = ""
        for line in all_out.split("\n"):
            ln = line.strip()
            if ("passed" in ln or "failed" in ln or "error" in ln) and any(c.isdigit() for c in ln):
                summary = ln
        if r.returncode == 0:
            print(f"PASS ({elapsed:.1f}s) {summary}")
            passed += 1
        else:
            print(f"FAIL ({elapsed:.1f}s) {summary}")
            for line in all_out.split("\n")[-5:]:
                if line.strip(): print(f"    {line.strip()}")
            failed += 1; fail_list.append(f)
    except subprocess.TimeoutExpired:
        elapsed = time.time()-t0
        print(f"TIMEOUT ({elapsed:.1f}s)")
        timeout_cnt += 1; timeout_list.append(f)
    except Exception as e:
        elapsed = time.time()-t0
        print(f"ERROR ({elapsed:.1f}s) {e}")
        err_cnt += 1; err_list.append(f)

print(f"\n{'='*60}")
print(f"TOTAL: {len(files)} files | PASS: {passed} | FAIL: {failed} | TIMEOUT: {timeout_cnt} | ERROR: {err_cnt}")
if fail_list: print("Failed:", ", ".join(fail_list))
if timeout_list: print("Timeout:", ", ".join(timeout_list))
if err_list: print("Error:", ", ".join(err_list))
