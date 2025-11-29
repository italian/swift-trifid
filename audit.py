import subprocess
import sys

def run_command(command, description):
    print(f"--- Running {description} ---")
    try:
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
        print(result.stdout)
        if result.stderr:
            print("Errors/Warnings:")
            print(result.stderr)
        return result.returncode
    except Exception as e:
        print(f"Failed to run {description}: {e}")
        return 1

def main():
    with open("audit_report.txt", "w", encoding="utf-8") as f:
        f.write("=== STARTING AUTOMATED AUDIT ===\n\n")

        files_to_check = "agent.py game.py model.py helper.py leaderboard.py play_human.py play_ai.py"

        # 1. Security Check (Bandit)
        f.write("[1/4] Security Audit (Bandit)\n")
        print("Running Bandit...")
        result = subprocess.run(f"{sys.executable} -m bandit -r {files_to_check}", capture_output=True, text=True, shell=True)
        f.write(result.stdout)
        if result.stderr:
            f.write("Errors/Warnings:\n")
            f.write(result.stderr)
        security_score = result.returncode

        # 2. Code Quality (Pylint)
        f.write("\n[2/4] Code Quality Audit (Pylint)\n")
        print("Running Pylint...")
        disabled_checks = "C0114,C0115,C0116,E1101,R0913,R0917,R0902,R0903,R0914"
        result = subprocess.run(f"{sys.executable} -m pylint {files_to_check} --disable={disabled_checks}",
                              capture_output=True, text=True, shell=True)
        f.write(result.stdout)
        quality_score = result.returncode

        # 3. Type Safety (Mypy)
        f.write("\n[3/4] Type Safety Audit (Mypy)\n")
        print("Running Mypy...")
        result = subprocess.run(f"{sys.executable} -m mypy {files_to_check}", capture_output=True, text=True, shell=True)
        f.write(result.stdout)
        type_score = result.returncode

        # 4. Logic Tests (Unit Tests)
        f.write("\n[4/4] Logic Tests (Unit Tests)\n")
        print("Running Unit Tests...")
        result = subprocess.run("python -m unittest test_logic -v",
                              capture_output=True, text=True, shell=True)
        f.write(result.stdout)
        if result.stderr:
            f.write(result.stderr)
        test_score = result.returncode

        f.write("\n=== AUDIT COMPLETE ===\n")
        if security_score == 0 and quality_score == 0 and type_score == 0 and test_score == 0:
            f.write("RESULT: PASSED. All checks passed.\n")
        else:
            f.write("RESULT: ISSUES FOUND. Please review the output above.\n")

    print("Audit complete. Report saved to audit_report.txt")

if __name__ == "__main__":
    main()
