import random, os.path as osp, re, numpy as np
from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from mmengine import load, dump

from prompts import prompt_fn
from utils.openai_utils import OpenAIDecodingArguments, get_total_tokens
from utils.model_utils import ChatBot
from utils.misc import lstrip_string, compute_metric, get_top_k_voting, compare_results

def parse_steps(model_output):
    lines = model_output.split("\n")
    lines = [line.strip() for line in lines if not (line.startswith("First, let's") or line.startswith("Next, let's")) and len(line.strip()) > 0]
    model_output = "\n".join(lines)
    blocks = re.split("\n#", model_output)
    steps = [block.lstrip("#") for block in blocks]
    reasoning_steps = []
    base_info = []
    if args.ref_end:
        reasoning_steps = [step for step in steps if "Reference" in step or "Step" in step]
    else:
        for step in steps:
            if "(by" in step or "Step" in step:
                reasoning_steps.append(step)
            else:
                base_info.append(step)
    return steps, reasoning_steps, "\n".join(base_info[:-1]) + "\n"


def verify_steps(steps, reasoning_steps):
    steps = deepcopy(steps)
    print(f"We have {len(reasoning_steps)} reasoning steps to be verified.")
    ret = []
    grounding_steps = []
    reasoning_steps_tmp = []
    reasoning_statements = []
    for i, reasoning_step in enumerate(reasoning_steps):
        try:
            step_idx = eval(reasoning_step.split(" ")[0].strip("#.: "))
            if args.ref_end:
                reasoning_step = lstrip_string(reasoning_step.strip(), f"{step_idx}.").strip(". ")
                statement = reasoning_step.split("Reference")[0].strip()
                grounding = reasoning_step.split("Reference")[1].strip(". :")
                for groups in re.findall("(Step \d+:)|(#\d+)", statement):
                    for group in groups:
                        statement = statement.replace(group, "").strip()
                statement = statement.replace(" . ", " ")
            else:
                reasoning_step = lstrip_string(reasoning_step, f"#{step_idx}.")
                statement = ")".join(reasoning_step.split(")")[1:]).strip()
                for group in re.findall("(Step \d+:)", statement):
                    statement = statement.replace(group, "").strip()

            steps[step_idx - 1] = statement
            if args.ref_end:
                grounding = grounding
            else:
                grounding = lstrip_string(re.findall("\((by[^)]+)\)", reasoning_step)[0], "by").strip()
            if "-" in grounding:
                grounding = grounding.split("-")
                assert len(grounding) == 2
                begin = grounding[0].strip(" #")
                end = grounding[1].strip(" #")
                grounding = list(range(eval(begin), eval(end) + 1))
            else:
                grounding = grounding.split("#")[1:]
                grounding = [eval(re.findall("\d{1,2}", _)[0]) for _ in grounding]

            grounding_steps += grounding
            reasoning_steps_tmp.append(step_idx)
            reasoning_statements.append(statement)

            if len(reasoning_step.strip().split("\n")) == 1 and i != len(reasoning_steps) - 1:
                ret.append(None)
                continue
            ground_materials = "\n".join([lstrip_string(steps[_ - 1], f"{_}. ").strip() for _ in grounding_steps if _ not in reasoning_steps_tmp])

            statement = "\n\n".join(reasoning_statements)

            prompt = f"""Here is some information:
"{ground_materials}"

Based on the given information, here is a reasoning process:
"{statement}"

Double-check the reasoning process, let's analyze its correctness, and end with "yes" or "no".

Answer:
Let's think step by step without any assumptions.
"""
            grounding_steps = []
            reasoning_steps_tmp = []
            reasoning_statements = []
            if args.verify_mode == "sequential":
                check_results = {}
                for check_mode in ["grounding", "reasoning", "calculation"]:
                    prompt = (
                        f'Here is some information:\n"{ground_materials}"\n\nBased on the given information, here is a reasoning process:\n"{statement}"'
                    )
                    prompt = prompt_fn(prompt, check_mode)
                    verify_model_outputs = call_model(prompt)
                    check_results.update({f"{check_mode}_check_outputs": verify_model_outputs, f"{check_mode}_check_inputs": prompt})
                    _, filtered_results = parse_results(verify_model_outputs)
                    if np.mean(filtered_results) < 0.5:
                        break
                ret.append(check_results)
            elif args.verify_mode == "simultaneous":
                prompt = prompt_fn(prompt, "verification")
                verify_model_outputs = call_model(prompt)
                ret.append({"verify_model_outputs": verify_model_outputs, "verify_model_inputs": prompt})
                _, filtered_results = parse_results(verify_model_outputs)
                if np.mean(filtered_results) < 0.5:
                    break
            else:
                raise NotImplementedError
        except Exception as e:
            print(f"Get an error when parsing steps! Error: {e}!")
            ret.append(None)
            continue
    return ret

def run_verify_naive(model_output, question):
    prompt = f'Here is a question and its solution:\n"Question:\n{question}\n\nAnswer:\n{model_output}"\n\nYou are a math teacher. Do you think the reasoning process is correct?\nLet\'s think step by step. End with "The reasoning process is".'
    return call_model(prompt)

def run_verify_cot(model_output, question):
    steps, reasoning_steps, _ = parse_steps(model_output)
    return verify_steps(steps, reasoning_steps)

def parse_results(verify_step):
    verify_model_outputs_filtered = []
    verify_model_outputs_raw = []
    for _ in verify_step:
        __ = [__ for __ in _.split("\n") if len(__) > 0][-1]
        verify_model_outputs_raw.append(__)
        if "not applicable" in __ or "N/A" in __ or "passes" in __:
            verify_answer = "yes"
        elif "fails" in __:
            verify_answer = "no"
        else:
            verify_answer = re.findall(
                '[ :\\"](Yes|No|yes|no|NO|YES|is correct or not|is incorrect|is not correct|partially correct|cannot determine|cannot confirm)[ ,\.\\"]',
                __,
            )
            if len(verify_answer) > 0:
                verify_answer = verify_answer[0].lower()
                verify_answer = (
                    "no"
                    if verify_answer
                    in [
                        "no",
                        "is not correct",
                        "is incorrect",
                        "is correct or not",
                        "partially correct",
                        "cannot determine",
                        "cannot confirm",
                    ]
                    else "yes"
                )
            else:
                verify_answer = re.findall('(Yes|No|yes|no|NO|YES)[ ,\.\\"]', __)
                verify_answer = "yes" if len(verify_answer) == 0 else verify_answer[0].lower()

        verify_model_outputs_filtered.append(verify_answer == "yes")
    return verify_model_outputs_raw, verify_model_outputs_filtered

def verification():
    random.seed(42)
    examples_be_skip = defaultdict(list)
    ## example would be skipped for wrong ground truth answer or extremely long reasoning steps(enumeration)
    examples_be_skip.update({"single_eq": [149, 209, 482], "aqua_rat": [213], "date": [297, 322], "MATH": [2874]})

    correct_in = 0
    if args.num_examples == -1 and args.task_name != "verify_step":
        ## verify questions with multi-majority answers by self-consistency
        error_examples = []
        correct_examples = []
        count_delta_size = defaultdict(int)
        majority_correct = 0
        for i, result_i in enumerate(input_result):
            deprecated_keys = [
                "majority_result",
            ]
            for key in deprecated_keys:
                if key in result_i:
                    result_i.pop(key)
            result_i["majority_correct"] = np.mean(result_i["majority_corrects"])
            pred_answers = result_i["pred_answers"]
            top2_voting = get_top_k_voting(pred_answers, k=2)
            result_i["sample_idx_need_verify"] = []

            sign = False
            for key in top2_voting:
                sign = sign or compare_results(key, result_i["final_answer"])
                result_i["sample_idx_need_verify"] += top2_voting[key]

            majority_correct += result_i["majority_correct"]
            if result_i["example_idx"] in examples_be_skip[data_name] or len(top2_voting) == 0:
                continue

            if len(top2_voting) >= 2 or len(pred_answers) == 1:
                keys = list(top2_voting.keys())

                min_key = keys[0]
                max_key = keys[-1]
                for key in top2_voting:
                    if len(top2_voting[key]) > len(top2_voting[max_key]):
                        max_key = key
                    if len(top2_voting[key]) < len(top2_voting[min_key]):
                        min_key = key
                min_size, max_size = len(top2_voting[min_key]), len(top2_voting[max_key])

                result_i["need_verify_samples_info"] = ", ".join([f"{key}->{len(item)}" for key, item in top2_voting.items()])
                if (min_size >= 2 and max_size - min_size <= 2) or len(pred_answers) == 1:
                    correct_in += sign
                    if result_i["majority_correct"] > 0:
                        correct_examples.append(result_i)
                    elif sign or len(pred_answers) == 1:
                        count_delta_size[max_size - min_size] += 1
                        error_examples.append(result_i)
        global_majority_correct = majority_correct
        print(f"#all samples: {len(input_result)}, #num majority correct: {majority_correct}, majority_correct acc: {majority_correct / len(input_result) * 100:.2f}%")
        print(
            f"#examples {len(correct_examples) + len(error_examples)}, #correct {len(correct_examples)}, #savable wrong {len(error_examples)}, #savable ratio {len(error_examples) / len(input_result) * 100:.2f}%"
        )
        count_delta_size = dict(count_delta_size)
        for key in sorted(count_delta_size.keys()):
            print(f"{key} -> {count_delta_size[key] / len(input_result) * 100:.2f}%")
        random.seed(42)
        random.shuffle(correct_examples)
        random.shuffle(error_examples)
        if args.wrong_only:
            examples = error_examples
        else:
            examples = correct_examples + error_examples
        random.shuffle(examples)
        results = []
    else:
        
        examples = input_result[: args.num_examples]
        random.shuffle(examples)
    existing_results = load(result_file) if osp.exists(result_file) else []
    existing_results = {result["example_idx"]: result for result in existing_results}

    print(f"We have tested {len(existing_results)} examples and totally have {len(examples)} examples.")

    num_call = count = count0 = count1 = num = num0 = num1 = 0
    wrong2right = right2wrong = 0
    results = []
    if args.task_name == "verify_cot":
        for i, example in enumerate(examples):
            example_idx = example["example_idx"]
            model_outputs = example["model_outputs"]
            final_answer = example["final_answer"]
            per_sample_result = example["pred_answers"]
            sample_idx_need_verify = example["sample_idx_need_verify"]
            question = example["question"]
            per_sample_correct = example["per_sample_correct"]
            majority_correct = example["majority_correct"]
            results_need_verify = [per_sample_result[sample_idx_need_verify[j]] for j in range(len(sample_idx_need_verify))]

            if example_idx not in existing_results:
                num_call += 1
                example["verify_model_results"] = []
            else:
                tmp = existing_results.pop(example_idx)
                tmp["majority_results"] = example["majority_results"]
                tmp["majority_correct"] = example["majority_correct"]
                example = tmp

            results.append(example)
            verify_model_results = example["verify_model_results"]

            print(results_need_verify, len(results_need_verify))
            print(
                f"Example idx:{example['example_idx']}, Pred: {example['majority_results']}, GT: {final_answer}, Correct: {majority_correct}, Verify Info: {example['need_verify_samples_info']}"
            )
            for j, sample_idx in enumerate(sample_idx_need_verify):
                if len(verify_model_results) <= j:
                    print(f"Check steps {j}.")
                    model_output = model_outputs[sample_idx]
                    if args.verify_mode == "naive":
                        verify_model_results.append(run_verify_naive(model_output, question))
                    else:
                        verify_model_results.append(run_verify_cot(model_output, question))
                    dump(list(existing_results.values()) + results, str(result_file), indent=4)

            example["verify_correct"] = []
            example["verify_result"] = []

            for sample_idx, sample_verify_result in zip(sample_idx_need_verify, verify_model_results):
                all_steps_results = []
                all_step_correct_raw = []
                if args.verify_mode == "naive":
                    correct_raw, parsed_outputs_filtered = parse_results(sample_verify_result)
                    all_steps_results.append(np.mean(parsed_outputs_filtered) > 0.5)
                    all_step_correct_raw.append(correct_raw)
                else:
                    for step_result in sample_verify_result:
                        if step_result is None:
                            all_steps_results.append(True)
                            all_step_correct_raw.append(None)
                            continue
                        if args.verify_mode == "sequential":
                            correct_raw = []
                            step_corrects = []
                            for k, v in step_result.items():
                                if "input" in k:
                                    continue
                                correct_raw_i, parsed_outputs_filtered = parse_results(v)
                                step_correct = np.mean(parsed_outputs_filtered) > 0.5
                                correct_raw.append(correct_raw_i)
                                step_corrects.append(step_correct)
                            step_correct = np.all(step_corrects)
                        else:
                            step_result = step_result["verify_model_outputs"]
                            correct_raw, parsed_outputs_filtered = parse_results(step_result)
                            step_correct = np.mean(parsed_outputs_filtered) > 0.5
                        all_steps_results.append(step_correct)
                        all_step_correct_raw.append(correct_raw)
                example["verify_correct"].append(np.all(all_steps_results))
                example["verify_result"].append(all_step_correct_raw)

                num += 1
                count += int(np.all(all_steps_results) == per_sample_correct[sample_idx])
                if per_sample_correct[sample_idx]:
                    count1 += int(np.all(all_steps_results) == per_sample_correct[sample_idx])
                    num1 += 1
                else:
                    count0 += int(np.all(all_steps_results) == per_sample_correct[sample_idx])
                    num0 += 1

            money = get_total_tokens() / 1000 * 0.002
            print(
                f"==> Acc {count / max(num, 1)}/[{count}/{num}], Acc0 {count0 / max(num0, 1)}/[{count0}/{num0}], Acc1 {count1 / max(num1, 1)}/[{count1}/{num1}], money={money:.4f}, ETM={money * (len(examples) - i - 1) / max(num_call, 1):.4f}"
            )

            print("Result need verify", results_need_verify, "Verified result", example["verify_correct"])
            pred_answers = [results_need_verify[j] for j in range(len(results_need_verify)) if example["verify_correct"][j]]
            if not len(per_sample_result):
                print("All the answers are verified to be wrong !!")
                continue

            per_sample_correct, majority_result, majority_corrects, majority_counts = compute_metric(pred_answers, final_answer)
            majority_correct = np.mean(majority_corrects)
            print(f"Results after filtering: {len(sample_idx_need_verify)} -> {len(pred_answers)}, {pred_answers}")

            print("New majority result:", majority_result, "New majority correct:", majority_correct)

            global_majority_correct = global_majority_correct - example["majority_correct"] + majority_correct
            if majority_correct > example["majority_correct"]:
                wrong2right += 1
            elif majority_correct < example["majority_correct"]:
                right2wrong += 1

            print(f"{i + 1} / {len(examples)}: wrong->right={wrong2right}, right->wrong={right2wrong}")
        dump(list(existing_results.values()) + results, str(result_file), indent=4)
        print(f"New majority correct: {global_majority_correct / len(input_result)}")
    elif args.task_name == "verify_step":
        for i, example in enumerate(examples):
            question = example["question"]
            pred_answer = example["answer"]
            final_answer = example["final_answer"]
            sample_correct = example["flag"]

            if len(existing_results) <= i:
                print("Check reasoning ......")
                if args.verify_mode == "naive":
                    verify_outputs = run_verify_naive(pred_answer, question)
                elif args.verify_mode == "simultaneous":
                    verify_outputs = run_verify_cot(pred_answer, question)
                else:
                    raise NotImplementedError
                example["verify_outputs"] = verify_outputs
                dump(list(existing_results.values()) + results, str(result_file), indent=4)
            else:
                example = existing_results[i]
                verify_outputs = example["verify_outputs"]

            results.append(example)
            example["verify_correct"] = []
            example["verify_result"] = []
            all_steps_results = []
            all_step_correct_raw = []
            if args.verify_mode == "naive":
                correct_raw, parsed_outputs_filtered = parse_results(verify_outputs)
                example["verify_correct"].append(np.mean(parsed_outputs_filtered) > 0.5)
                example["verify_result"].append(correct_raw)
            else:
                for step_result in verify_outputs:
                    if step_result is None:
                        all_steps_results.append(True)
                        all_step_correct_raw.append(None)
                        continue
                    if args.verify_mode == "sequential":
                        correct_raw = []
                        step_corrects = []
                        for k, v in step_result.items():
                            if "input" in k:
                                continue
                            correct_raw_i, parsed_outputs_filtered = parse_results(v)
                            step_correct = np.mean(parsed_outputs_filtered) > 0.5
                            correct_raw.append(correct_raw_i)
                            step_corrects.append(step_correct)
                        step_correct = np.all(step_corrects)                                 
                    else:
                        step_result = step_result["verify_model_outputs"]
                        correct_raw, parsed_outputs_filtered = parse_results(step_result)
                        step_correct = np.mean(parsed_outputs_filtered) > 0.5
                    all_steps_results.append(step_correct)
                    all_step_correct_raw.append(correct_raw)
                example["verify_correct"].append(np.all(all_steps_results))
                example["verify_result"].append(all_step_correct_raw)

            num += 1
            count += int(example["verify_correct"] == sample_correct)
            if sample_correct:
                count1 += int(example["verify_correct"] == sample_correct)
                num1 += 1
            else:
                count0 += int(example["verify_correct"] == sample_correct)
                num0 += 1

            money = get_total_tokens() / 1000 * 0.002
            print(
                f"==> Acc {count / max(num, 1)}/[{count}/{num}], Acc0 {count0 / max(num0, 1)}/[{count0}/{num0}], Acc1 {count1 / max(num1, 1)}/[{count1}/{num1}], money={money:.4f}, ETM={money * (len(examples) - i - 1) / max(num_call, 1):.4f}"
            )

            print("Verified result", example["verify_correct"])
            dump(list(existing_results.values()) + results, str(result_file), indent=4)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate Verification Performance")
    parser.add_argument("--model-name", default="gpt3", type=str)
    parser.add_argument("--data-name", default="gsm8k", type=str)
    parser.add_argument("--input-result", default="/tmp/input.json", type=str)
    parser.add_argument("--output-result", default="/tmp/result.json", type=str)
    parser.add_argument("--num-examples", default=-1, type=int)
    parser.add_argument("--wrong-only", default=False, type=bool)
    parser.add_argument("--max-seq-len", default=2048, type=int)
    parser.add_argument("--task-name", default="verify_cot", type=str)
    parser.add_argument("--verify-mode", default="simultaneous", type=str)
    parser.add_argument("--ref-end", default=False, type=bool)
    parser.add_argument("--tag", default=None, type=str)
    parser.add_argument("--n", default=3, type=int)
    parser.add_argument("--greedy", action="store_true", default=False)
    args = parser.parse_args()

    ## naive: zero-shot prompt to verify the whole process,
    ## simultaneous: one-shot prompt to verify each steps, all at once
    ## sequential: one-shot prompt to verify each steps, split checks
    assert args.verify_mode in ["naive", "simultaneous", "sequential"], f"{args.verify_mode} mode is not supported!"
    data_name, model_name = args.data_name, args.model_name
    model_print_name = (f"chat-{model_name}" if model_name in ChatBot.MODEL_TYPE else model_name) if args.tag is None else args.tag
    input_result = load(args.input_result)

    result_file = Path(args.output_result)
    results = load(result_file) if result_file.exists() else []
    num_call = 0

    sample_n = 1 if args.greedy else args.n
    temperature = 0 if args.greedy else 0.7
    decoding_args = OpenAIDecodingArguments(max_tokens=args.max_seq_len, n=sample_n, temperature=temperature)

    print(f"The results are saved in {str(result_file)}. Take name: {args.task_name}. Dataset: {data_name}")

    def call_model(prompt):
        return ChatBot.call_chat_gpt(
            prompt,
            eos_pattern=None,
            max_new_tokens=args.max_seq_len,
            early_stopping=True,
            do_sample=not args.greedy,
            return_list=True,
            temperature=temperature,
            num_beams=sample_n,
            num_return_sequences=sample_n,
            decoding_args=decoding_args,
        )

    ChatBot.dataset_name = data_name
    ChatBot.init(model_name)
    verification()
