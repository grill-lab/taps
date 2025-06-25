import pandas as pd
import os
import json
import re
import numpy as np
from copy import deepcopy
from datasets import Dataset
from collections import defaultdict
from copy import deepcopy

import matplotlib.pyplot as plt


def get_targets(targets_dir, return_instructions=False, split="test"):
    fpath = os.path.join(targets_dir, split + ".jsonl")
    targets = pd.read_json(fpath, lines=True)

    api_calls = targets["api_calls"]

    if return_instructions:
        instr = targets["applicable_standing_instructions"]
        return api_calls, instr
    return api_calls


def get_selection_data(data: pd.DataFrame) -> tuple:
    """
    Returns a list of lists of selections, and a list of lists of gold selections
    """
    gold_selections = (
        data["applicable_standing_instructions"]
        .apply(lambda x: [i["nl_instruction"].strip(" .") for i in x])
        .tolist()
    )
    profiles = (
        data["all_standing_instructions"]
        .apply(lambda x: [i["nl_instruction"].strip(" .") for i in x])
        .tolist()
    )
    example_ids = data["example_id"].tolist()

    domains = []
    known_domains = (
        data["applicable_standing_instructions"]
        .apply(lambda x: [i["instruction_src"] for i in x])
        .tolist()
    )
    for p, profile in enumerate(profiles):
        prof_domains = []
        for instr in profile:
            if instr in gold_selections[p]:
                ind = gold_selections[p].index(instr)
                prof_domains.append(known_domains[p][ind])
            else:
                prof_domains.append("NR")
        domains.append(prof_domains)
    return gold_selections, profiles, example_ids, domains


def convert_domain_kv_to_array(domain: str, key_value_pairs: dict) -> str:
    arr = []
    for key, value in key_value_pairs.items():
        s = domain.lower() + "_" + key.lower() + "=" + str(value).lower()
        arr.append(s)
    return arr


def simplify_names(kv: dict, alt_names: dict):
    kv = deepcopy(kv)
    for key, val in kv.items():
        kv[key] = alt_names.get(val.lower(), val)
    return kv


def get_api_data(data: Dataset, predictions: list, alt_names_path: str) -> tuple:
    """
    Returns a list of lists of slots, and a list of lists of gold slots
    """
    predicted_slots = []
    gold_slots = []
    example_ids = []
    lengths = []

    alt_names = load_json(alt_names_path)

    for datum, preds in zip(data, predictions):
        example_ids.append(datum["example_id"])
        curr_gold_slots = []
        curr_gold_domain = defaultdict(list)
        gold_length = len(datum["api_calls"])
        versus = None
        for api in datum["api_calls"]:
            domain, kv = get_domain_kv(api)
            kv = simplify_names(kv, alt_names)
            for k, v in kv.items():
                v = re.sub("(\W)versus(\W)", "\g<1>vs\g<2>", v)
                if v == "any":
                    continue
                curr_gold_domain[domain].append({k: v})
            curr_gold_slots = curr_gold_slots + convert_domain_kv_to_array(domain, kv)
        gold_slots.append(curr_gold_slots)

        curr_predicted_slots = []
        c = 0
        for api in preds.splitlines():
            if api.startswith("Get"):
                domain, kv = get_domain_kv(api)
                kv = simplify_names(kv, alt_names)

                ## handling corner cases
                category_flag = False
                event_type_flag = False
                if domain in curr_gold_domain:
                    kv_list = curr_gold_domain[domain]
                    curr_keys = [list(k.keys())[0] for k in kv_list]
                    for k, v in kv.items():
                        kv[k] = re.sub("(\W)versus(\W)", "\g<1>vs\g<2>", v)
                        if "time" in k or "date" in k:
                            if k in curr_keys:
                                idx = curr_keys.index(k)
                                gold_value = curr_gold_domain[domain][idx][k].lower()

                                if v.lower() == gold_value:
                                    continue
                                value_tokens = v.lower().split()
                                if len(value_tokens) == 1:
                                    continue

                                gold_value_tokens = gold_value.split()

                                # difference between gold_value_tokens and tokens
                                diff = set(gold_value_tokens) - set(value_tokens)

                                if len(diff) == 1:
                                    kv[k] = gold_value
                                if v.lower() in gold_value.lower():
                                    kv[k] = gold_value

                        if k == "event_type" and "subcategory" not in curr_keys:
                            if (
                                "category" in curr_keys
                                and "event_type" not in curr_keys
                            ):
                                category_flag = True
                        if k == "category" and "subcategory" not in curr_keys:
                            if (
                                "event_type" in curr_keys
                                and "category" not in curr_keys
                            ):
                                event_type_flag = True
                if category_flag:
                    v = kv["event_type"]
                    kv["category"] = v
                    del kv["event_type"]

                if event_type_flag:
                    v = kv["category"]
                    kv["event_type"] = v
                    del kv["category"]

                new_kv = deepcopy(kv)
                for k, v in kv.items():
                    if v == "any":
                        del new_kv[k]
                kv = new_kv

                curr_predicted_slots = (
                    curr_predicted_slots + convert_domain_kv_to_array(domain, kv)
                )
                c += 1
        lengths.append(c != gold_length)

        predicted_slots.append(curr_predicted_slots)
    return gold_slots, predicted_slots, example_ids, lengths


def postprocess_preds(preds):
    processed = []
    for pred in preds:
        pred = pred.split("[INST]")[0].split("']")[0]
        pred = pred.replace("Get", "\nGet")
        pred = re.sub("(Get[A-z]+?) \(", "\g<1>\(", pred)
        pred = re.sub(r"\\", "", pred)
        pred = [l.strip() for l in pred.splitlines() if l.startswith("Get")]
        processed.append("\n".join(pred))
    return processed


def get_domain_kv(input_string: str) -> tuple:
    domain_pattern = r"Get(\w+)\((.+)\)"  # defined as our API call
    matches = re.findall(domain_pattern, input_string)
    domain = ""
    if matches:
        domain = matches[0][0]
    if "True" in input_string:
        idx = input_string.index("True")
        if input_string[idx - 1] != '"':
            input_string = input_string.replace("True", '"True"')
    input_string = re.sub("=([1-9].*?)(,|\))", '="\g<1>"\g<2>', input_string)
    pattern = r'(\w+)="([^"]+)"'

    matches = re.findall(pattern, input_string)
    if not matches:
        return domain, {}

    key_value_pairs = {}
    for match in matches:
        key = match[0]
        value = match[1]
        key_value_pairs[key] = value

    return domain, key_value_pairs


def load_json(fpath):
    with open(fpath, encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_interpretation_results(res_dir, setup, model, shots, formatted=False):

    if formatted:
        dir_path = os.path.join(res_dir, formatted, setup, model, str(shots))

        # get predictions
        preds_fname = (
            "test.predictions-up.jsonl"
            if "test.predictions-up.jsonl" in os.listdir(dir_path)
            else "test.predictions.jsonl"
        )
        preds_fpath = os.path.join(dir_path, preds_fname)
        preds = pd.read_json(preds_fpath, lines=True)
        preds = preds.drop(
            columns=[
                "slot_values_update",
                "system_frame",
                "prev_active_intent",
                "system_utterance",
                "prev_user_utterance",
            ]
        )

        # get aggregated scores
        agg_scores_fpath = os.path.join(dir_path, "test.metrics.aggregate_metrics.json")
        agg_scores = load_json(agg_scores_fpath)

        # get instance-wise scores
        inst_scores_fpath = os.path.join(
            dir_path, "test.metrics.instance_wise_metrics.json"
        )
        inst_scores = load_json(inst_scores_fpath)

        # get generation parameters
        fpath = os.path.join(
            res_dir, "raw", setup, model, f"{model}__{shots}__preds.json"
        )
        results = load_json(fpath)

        results["formatted_preds"] = preds
        #         metrics = {'agg_metrics': agg_scores, 'instance_metrics': inst_scores}
        #         results.update({'metrics': metrics})
        results.update(agg_scores)
        results.update(inst_scores)

    else:
        fpath = os.path.join(res_dir, setup, model, f"{model}__{shots}__preds.json")
        results = load_json(fpath)

    return results


def get_triples_func(func):

    triples = []

    org_func = func
    if not func.endswith(")"):
        return None

    # convert to lowercase
    func = func.lower()
    func = re.sub("""=[^"'](.+?)(,|\))""", '="\g<1>"\g<2>', func)

    # unify 'vs' and 'versus'
    func = re.sub("vs(\W|\b|$)", "versus\g<1>", func)

    # unify quotation marks
    func = re.sub(r"""[“"”‘’«»„“„”»«']([^s])""", str("\u0022") + "\g<1>", func)

    # unify 'subcategory' and 'event_type'
    func = re.sub("event_type=", "category=", func)
    func = re.sub("subcategory=", "category=", func)

    func_name, args = func.split("(", 1)
    func_name = func_name.strip()

    args = args.split('", ')
    for arg in args:
        slot, value = arg.split("=")
        value = value.strip(" )")
        if not value.endswith('"'):
            value = value + '"'
        triples.append((func_name, slot.strip(), value.strip('"')))

    return triples


def get_triples(preds):
    all_triples = []
    unfinished = 0

    for pred in preds:
        triples = []
        for func in pred:
            triple = get_triples_func(func)
            if triple is not None:
                triples.append(triple)
            else:
                unfinished += 1
        all_triples.append(triples)
    return all_triples, unfinished


def get_subset_example(pred, target, subset="func"):
    transp = {"func": 0, "slot": 1, "val": 2}
    if subset == "func":
        pred_funcs = [func[0][transp[subset]] for func in pred]
        target_funcs = [func[0][transp[subset]] for func in target]
    else:
        pred_funcs = [
            [(slot[0], slot[transp[subset]]) for slot in func] for func in pred
        ]
        target_funcs = [
            [(slot[0], slot[transp[subset]]) for slot in func] for func in target
        ]

    return pred_funcs, target_funcs


def get_subsets(preds, targets, subset):
    pred_data, target_data = [], []
    for p, t in zip(preds, targets):

        p, t = get_subset_example(p, t, subset)
        pred_data.append(p)
        target_data.append(t)
    return pred_data, target_data


def em_example(pred, target):
    if len(pred) == 0 and len(target) != 0:
        return 0
    if len(pred) == 0 and len(target) == 0:
        return 1
    if not isinstance(pred[0], str):
        pred = sum(pred, [])
        target = sum(target, [])
    return int(sorted(pred) == sorted(target))


def update_doubles(example):

    if len(set(example)) == len(example):
        return example

    counter = {}
    upd_example = []
    for i in example:
        if i in upd_example:
            if isinstance(i, str):
                ii = i
            else:
                ii = i[0]
            ind = counter.get(i, 0) + 1
            counter[i] = ind
            if isinstance(i, str):
                i = i + f"_{ind}"
            else:
                i = (i[0] + f"_{ind}", i[1])
        upd_example.append(i)
    return upd_example


def f1_example(pred, target):

    if not isinstance(pred[0], str):
        pred = sum(pred, [])
        target = sum(target, [])

    #     pred = update_doubles(pred)
    #     target = update_doubles(target)

    tp = len(set(pred) & set(target))
    if tp == 0:
        return 0, 0, 0

    precision = tp / len(set(pred))
    recall = tp / len(set(target))

    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def f1_score(preds, targets):
    f1s, precs, recs = [], [], []

    for pred, target in zip(preds, targets):
        if len(pred) == 0 and len(targets) != 0:
            f1, prec, rec = 0, 0, 0
        else:
            f1, prec, rec = f1_example(pred, target)
        f1s.append(f1)
        precs.append(prec)
        recs.append(rec)
    return np.mean(f1s), np.mean(precs), np.mean(recs), f1s, precs, recs


def analyse_interpretation_results(preds, targets):

    avg_num_calls_pred = np.mean([len(c) for c in preds])
    avg_num_calls_target = np.mean([len(c) for c in targets])
    overgenerates = sum(len(p) > len(t) for (p, t) in zip(preds, targets)) / len(
        targets
    )
    undergenerates = sum(len(p) < len(t) for (p, t) in zip(preds, targets)) / len(
        targets
    )

    pred_triples, unfinished = get_triples(preds)
    target_triples, _ = get_triples(targets)

    unfinished = unfinished / len(targets)

    results = {
        "avg_num_calls_pred": avg_num_calls_pred,
        "avg_num_calls_target": avg_num_calls_target,
        "overgeneration_percent": overgenerates,
        "undergeneration_percent": undergenerates,
        "unfinished": unfinished,
    }

    ems = [em_example(p, t) for (p, t) in zip(pred_triples, target_triples)]
    em = np.mean(ems)
    f1, precision, recall, f1_inst, prec_inst, rec_inst = f1_score(
        pred_triples, target_triples
    )
    results["em"] = em
    results["f1"] = f1
    results["precision"] = precision
    results["recall"] = recall
    results["instance_em"] = ems
    results["instance_f1"] = f1_inst
    results["instance_precision"] = prec_inst
    results["instance_recall"] = rec_inst

    for subset in ["func", "slot", "val"]:
        pred_data, target_data = get_subsets(
            pred_triples, target_triples, subset=subset
        )
        ems = [em_example(p, t) for (p, t) in zip(pred_data, target_data)]
        em = np.mean(ems)
        f1, precision, recall, f1_inst, prec_inst, rec_inst = f1_score(
            pred_data, target_data
        )

        results[subset + "_em"] = em
        results[subset + "_f1"] = f1
        results[subset + "_precision"] = precision
        results[subset + "_recall"] = recall

        results[subset + "_instance_em"] = ems
        results[subset + "_instance_f1"] = f1_inst

    return results


def plot_bar(em_scores, f1_scores, feature, title, average=False):
    em_counter = {}
    f1_counter = {}
    counter = {}

    for em, f1, feat in zip(em_scores, f1_scores, feature):
        if feat not in counter:
            counter[feat] = 0
            if average:
                em_counter[feat] = []
                f1_counter[feat] = []
            else:
                em_counter[feat] = 0
                f1_counter[feat] = 0
        if average:
            em_counter[feat].append(em)
            f1_counter[feat].append(f1)
        else:
            f1_counter[feat] += int(f1 <= 0.5)
            em_counter[feat] += int(em == 0)
        counter[feat] += 1

    if average:
        for key in counter:
            f1_counter[key] = np.mean(f1_counter[key])
            em_counter[key] = np.mean(em_counter[key])

    else:
        num_app = feature.value_counts().to_dict()
        for intent, val in em_counter.items():
            em_counter[intent] = val / num_app[intent]
        for intent, val in f1_counter.items():
            f1_counter[intent] = val / num_app[intent]

    f1_counter = pd.DataFrame([f1_counter]).T
    f1_counter = f1_counter.reset_index()

    em_counter = pd.DataFrame([em_counter]).T
    em_counter = em_counter.reset_index()

    scores = f1_counter.merge(em_counter, on="index")
    scores = scores.rename(columns={"0_x": "f1", "0_y": "em", "index": "category"})
    scores["percentage"] = scores["category"].apply(
        lambda x: round((counter[x] / len(em_scores)) * 100, 1)
    )
    scores = scores.sort_values("percentage", ascending=False)
    scores["category"] = scores.apply(
        lambda x: f"({x['percentage']}) {x['category']}", axis=1
    )
    scores = scores.drop(columns=["percentage"])

    scores = scores.sort_values("f1")
    #     scores = scores.sort_values('category')
    scores.plot(
        x="category",
        kind="barh",
        stacked=False,
        figsize=(10, 10),
        width=0.6,
    )
    plt.title(title)


#     return counter


def print_metrics(res):
    print("Overgeneration:", res["overgeneration_percent"] * 100)
    print("Undergeneration:", res["undergeneration_percent"] * 100)
    print()
    print("EM:", res["em"] * 100)
    print("F1:", res["f1"] * 100)
    print()
    print("Func F1:", res["func_f1"] * 100)
    print("Func EM:", res["func_em"] * 100)
    print()
    print("Slot F1:", res["slot_f1"] * 100)
    print("Slot EM:", res["slot_em"] * 100)
    print()
    print("Val F1:", res["val_f1"] * 100)
    print("Val EM:", res["val_em"] * 100)
