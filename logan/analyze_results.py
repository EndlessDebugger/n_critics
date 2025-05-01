from human_eval.data import stream_jsonl, write_jsonl
import os 

results_dir = os.path.join(os.path.abspath(__file__), '..', "results")
iter0_file = os.path.join(results_dir, "iter_0.jsonl_results.jsonl")
iter1_file = os.path.join(results_dir, "iter_1.jsonl_results.jsonl")

orig_better = []
ncritics_better = []

for iter0, iter1 in zip(stream_jsonl(iter0_file), stream_jsonl(iter1_file)):
    if iter0["passed"] and not iter1["passed"]: 
        orig_better.append(iter0)
        orig_better.append(iter1)

    elif iter1["passed"] and not iter0["passed"]: 
        ncritics_better.append(iter0)
        ncritics_better.append(iter1)

write_jsonl(os.path.join(results_dir, "orig_better.jsonl"), orig_better)
write_jsonl(os.path.join(results_dir, "ncritics_better.jsonl"), ncritics_better)
