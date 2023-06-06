import re, numpy as np
from numbers import Number
from collections import defaultdict


NEGATIVE_PATTERNS = [
    "is not needed to answer the question",
]


ANSWER_SPLIT_PATTERNS = [
    "answer is",
    "final answer:",
    "answer to the question is",
    "answer to this question is",
    "concatenated letters are",
    "concatenate the letters -",
    "The answer of ",
]

ANSWER_PREFIX = [
    "answer: ",
    "Therefore, there will be ",
    "Therefore, \w+ have ",
    "Therefore, \w+ and \w+ have ",
    "Therefore, \w+ has ",
    "Therefore,(.*?)is ",
    "answer to(.*?)is ",
    "answer to(.*?)will be ",
    "answer to(.*?)would be ",
    "answer to(.*?)becomes ",
    "Therefore,(.*?)will be ",
    "Therefore,(.*?)would be ",
    "Therefore,(.*?)cost ",
    "Therefore,(.*?)costs ",
    "Therefore,(.*?)a total of ",
    "There will be ",
    "Therefore, ",
    "[A-Z]\w+ will have ",
    "[A-Z]\w+ have ",
    "[A-Z]\w+ has ",
    "\w+ still has ",
    "^[A-Z]\w+ \w+ ",
]

NO_ANSWER_TEMPLATE = [
    "we cannot provide an answer to (this|the) question with the given information",
    "we cannot answer (this|the) question",
    "we cannot determine",
    "we can't determine",
    "we do not have enough information to answer (this|the) question",
    "we do not have enough information to provide a definitive answer to (this|the) question",
    "the answer(.*?)is unknown",
    "answer is not listed among the answer choices",
]

SKIP_ANSWER_TEMPLATE = [
    "Code cannot be executed!",
    "Code contains infinite loop!",
    "No answer!",
]

ZERO_ANSWER_TEMPLATE = [
    "doesn't have any money left",
    "used up all of",
]


NUM_RE_TEMPLATE = "(-?\d[\d,\. ]*)"
FRACTION_RE_TEMPLATE = "(-?\(\d+\/\d+\)\/\d+|-?\d+\/\d+)"
DATA_RE_TEMPLATE = "(\d\d\/\d\d\/\d\d\d\d)"
YES_NO_RE_TEMPLATE = "(?:Yes|No|yes|no|NO|YES)"

def lstrip_string(string, prefix):
    if string.startswith(prefix):
        return string[len(prefix) :]
    else:
        return string

def compute_metric(pred_answers, final_answer):
    majority_results, majority_count = majority_voting(pred_answers)

    per_sample_correct = [compare_results(pred_answer, final_answer) for pred_answer in pred_answers]
    majority_corrects = compare_results(majority_results, final_answer)
    return per_sample_correct, majority_results, majority_corrects, majority_count

def majority_voting(answers):
    count = count_answers(answers)
    valid_answer_count = [len(value) for key, value in count.items() if key not in SKIP_ANSWER_TEMPLATE]
    if len(valid_answer_count) > 0:
        majority_count = max(valid_answer_count)
        majority_results = [key for key in count if len(count[key]) == majority_count and key not in SKIP_ANSWER_TEMPLATE]
        return majority_results, majority_count
    else:
        return ["No answer!"], len(answers)

def compare_results(answers, final_answer):
    if len(re.findall(rf"^{DATA_RE_TEMPLATE}$", f"{final_answer}")) == 0:
        try:
            final_answer = eval(final_answer)
        except Exception as e:
            pass
    if not isinstance(answers, list):
        return compare_results([answers], final_answer)[0]
    if isinstance(final_answer, Number):
        try:
            ret = [np.abs(eval(answer) - final_answer) < 1e-6 for answer in answers]
        except Exception as e:
            ret = [answer == f"{final_answer}" for answer in answers]
    else:
        ret = [answer == f"{final_answer}" for answer in answers]
    return ret

def count_answers(answers):
    count = defaultdict(list)
    is_number = False
    for answer in answers:
        try:
            eval(answer)
            is_number = True
            break
        except Exception as e:
            pass
    if is_number:
        for i, answer in enumerate(answers):
            if answer in count:
                count[answer].append(i)
                continue

            found = False
            for key in count:
                try:
                    eval(key)
                except Exception as e:
                    continue

                try:
                    eval(answer)
                except Exception as e:
                    continue
                try:
                    if np.abs(eval(answer) - eval(key)) < 1e-6:
                        count[key].append(i)
                        found = True
                        break
                except Exception as e:
                    pass

            if not found:
                count[answer].append(i)
    else:
        for i, answer in enumerate(answers):
            count[answer].append(i)
    return count

def get_top_k_voting(answers, k=2):
    count = count_answers(answers)

    count_key = [key for key in count if key != "No answer!"]
    count_value = [len(count[key]) for key in count_key]
    sorted_idx = np.argsort(count_value)
    begin = len(sorted_idx) - k
    while begin > 0:
        if count_value[sorted_idx[begin]] != count_value[sorted_idx[begin - 1]]:
            break
        begin -= 1
    return {count_key[i]: count[count_key[i]] for i in sorted_idx[begin:]}