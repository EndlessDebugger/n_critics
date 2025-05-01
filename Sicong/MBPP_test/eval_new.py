# -*- coding: utf-8 -*-
"""
Evaluation script that runs tests against cleaned model outputs with updated test cases.
1. Load original MBPP dataset for setup code
2. Load cleaned model outputs (code + patched test_list)
3. Execute and evaluate each task
"""
import json
import traceback
import signal

# --- Timeout Handling ---
class TimeoutException(Exception):
    pass

def _handle_timeout(signum, frame):
    raise TimeoutException()

# 1. Load original MBPP for setup_code
def load_mbpp(path: str) -> dict:
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            tid = str(entry.get('task_id'))
            data[tid] = entry
    return data

# 2. Load cleaned model output with updated tests
def load_cleaned(path: str) -> dict:
    results = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            tid = str(entry.get('task_id'))
            results[tid] = {
                'completion': entry.get('completion', ''),
                'test_list':  entry.get('test_list', [])
            }
    return results

# 3. Run a single task's tests
def run_test(code_str: str, setup_code: str, tests: list, timeout: int = 5) -> bool:
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.alarm(timeout)
    try:
        local_env = {'__name__': '__main__'}
        exec(setup_code, local_env)
        exec(code_str, local_env)
        for test in tests:
            exec(test, local_env)
        signal.alarm(0)
        return True
    except TimeoutException:
        print(f"[Test failed] Timeout after {timeout}s")
        return False
    except Exception as e:
        print(f"[Test failed] {e}")
        traceback.print_exc()
        return False
    finally:
        signal.alarm(0)

# 4. Evaluate all tasks
def evaluate(mbpp_path: str, cleaned_path: str):
    mbpp_data = load_mbpp(mbpp_path)
    cleaned   = load_cleaned(cleaned_path)

    total, passed = 0, 0
    failed_cases = []

    for tid, rec in cleaned.items():
        code  = rec['completion']
        tests = rec['test_list']
        if not tests:
            continue

        total += 1
        print(f"Evaluating task {tid}")
        setup = mbpp_data.get(tid, {}).get('test_setup_code', '')
        success = run_test(code, setup, tests)
        if success:
            passed += 1
        else:
            failed_cases.append(tid)

    print("\nEvaluation complete.")
    print(f"Total evaluated: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Accuracy: {passed / total * 100:.2f}%")
    if failed_cases:
        print("Failed task_ids:", failed_cases)

if __name__ == '__main__':
    MBPP_PATH    = "/scratch/user/lsc206573/nlp/mbpp/mbpp.jsonl"
    CLEANED_PATH = "/scratch/user/lsc206573/nlp/mbpp/cleaned_samples.jsonl"
    evaluate(MBPP_PATH, CLEANED_PATH)

