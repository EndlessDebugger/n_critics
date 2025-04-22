import json
import traceback

# 1. load dataset
def load_mbpp(path):
    data = {}
    with open(path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            data[str(entry['task_id'])] = entry
    return data

# 2. load model output
def load_generated(path):
    results = {}
    with open(path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            results[str(entry['task_id'])] = entry['completion']
    return results

# 3. test code
def run_test(code_str, setup_code, test_cases):
    try:
        local_env = {}
        exec(setup_code, local_env)
        exec(code_str, local_env)
        for test in test_cases:
            exec(test, local_env)
        return True
    except Exception as e:
        print(f"[Test failed] {e}")
        traceback.print_exc()
        return False

# 4. evaluation function
def evaluate(mbpp_path, generated_path):
    mbpp = load_mbpp(mbpp_path)
    generated = load_generated(generated_path)

    total = 0
    passed = 0
    failed_cases = []

    for task_id, code in generated.items():
        if task_id not in mbpp:
            print(f"Warning: task_id {task_id} not in MBPP dataset.")
            continue

        entry = mbpp[task_id]
        setup = entry.get("test_setup_code", "")
        tests = entry.get("test_list", [])

        if not tests:
            continue

        total += 1
        success = run_test(code, setup, tests)
        if success:
            passed += 1
        else:
            failed_cases.append(task_id)

    print(f"\nEvaluation complete.")
    print(f" Total evaluated: {total}")
    print(f" Passed: {passed}")
    print(f" Failed: {total - passed}")
    print(f" Accuracy: {passed / total * 100:.2f}%")

    if failed_cases:
        print("\nFailed task_ids:", failed_cases)


if __name__ == "__main__":
    mbpp_path = "/scratch/user/lsc206573/nlp/mbpp/mbpp.jsonl"
    generated_path = "/scratch/user/lsc206573/nlp/mbpp/generated_samples.jsonl"
    evaluate(mbpp_path, generated_path)

