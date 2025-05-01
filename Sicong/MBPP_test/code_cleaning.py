# -*- coding: utf-8 -*-

import re
import json
from typing import List, Dict

# --- Configuration ---
MBPP_PATH    = "mbpp.jsonl"
REFINED_PATH = "refined_samples.jsonl"
CLEANED_PATH = "cleaned_samples.jsonl"


def clean_and_fix_signature(raw_code: str, tests: List[str]) -> str:

    # 1. Strip Markdown code blocks and backticks
    code = re.sub(r'```.*?```', '', raw_code, flags=re.DOTALL)
    code = code.replace('`', '')

    # 2. Collect valid code lines
    cleaned_lines: List[str] = []
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append(line)
            continue
        if re.match(r'^(import|from)\s+', stripped):
            cleaned_lines.append(line)
            continue
        if re.match(r'^[A-Za-z_]\w*\s*=', stripped):
            cleaned_lines.append(line)
            continue
        if stripped.startswith('#') or stripped.startswith('@'):
            cleaned_lines.append(line)
            continue
        if re.match(r'^(def|class)\s+', stripped):
            cleaned_lines.append(line)
            continue
        if line.startswith((' ', '\t')):
            cleaned_lines.append(line)
            continue
        # stop at first explanatory line
        break

    code_body = "\n".join(cleaned_lines)

    # 3. Infer function name from first test
    func_name = None
    if tests:
        m = re.search(r'(\w+)\s*\(', tests[0])
        if m:
            func_name = m.group(1)
    if not func_name:
        return code_body

    # 4. Extract full argument string (handles nested parentheses)
    call = tests[0]
    start = call.find(f"{func_name}(")
    i = start + len(func_name) + 1
    level = 1
    arg_chars: List[str] = []
    while i < len(call) and level > 0:
        c = call[i]
        if c == '(': level += 1; arg_chars.append(c)
        elif c == ')': level -= 1; \
arg_chars.append(c) if level > 0 else None
        else: arg_chars.append(c)
        i += 1
    full_args = ''.join(arg_chars).strip()

    # 5. Split on top-level commas to get parameters
    args, cur, lvl = [], [], 0
    for ch in full_args:
        if ch == '(':
            lvl += 1; cur.append(ch)
        elif ch == ')':
            lvl -= 1; cur.append(ch)
        elif ch == ',' and lvl == 0:
            args.append(''.join(cur).strip()); cur = []
        else:
            cur.append(ch)
    if cur: args.append(''.join(cur).strip())
    param_names = [f"param{i+1}" for i in range(len(args))]

    # 6. Remove original signature and leading blanks
    body_lines = code_body.splitlines()
    if body_lines and re.match(r'^\s*def\s+\w+\s*\(.*?\)\s*:', body_lines[0]):
        body_lines.pop(0)
    while body_lines and not body_lines[0].strip():
        body_lines.pop(0)

    # 7. Build new signature and preserve body indentation
    signature = f"def {func_name}({', '.join(param_names)}):"
    return "\n".join([signature] + body_lines)


def load_mbpp_tests(path: str) -> Dict[str, List[str]]:

    tests_map: Dict[str, List[str]] = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            tid = str(entry.get('task_id'))
            tests_map[tid] = entry.get('test_list', []) or []
    return tests_map


def main():
    mbpp_tests = load_mbpp_tests(MBPP_PATH)
    with open(REFINED_PATH, 'r', encoding='utf-8') as fin, \
         open(CLEANED_PATH, 'w', encoding='utf-8') as fout:
        for line in fin:
            entry = json.loads(line)
            tid   = str(entry.get('task_id'))
            raw   = entry.get('completion', '')
            tests = mbpp_tests.get(tid, [])

            # Clean and fix code signature
            cleaned_code = clean_and_fix_signature(raw, tests)

            # Replace old function name in test cases with new one
            old_name = re.search(r'(\w+)\s*\(', tests[0]).group(1) if tests else None
            new_name = re.match(r'\s*def\s+(\w+)\s*\(', cleaned_code)
            new_name = new_name.group(1) if new_name else old_name
            updated_tests = [t.replace(old_name, new_name) for t in tests] if old_name else tests

            entry['completion'] = cleaned_code
            entry['test_list']  = updated_tests
            fout.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"Cleaned outputs saved to {CLEANED_PATH}")


if __name__ == '__main__':
    main()
