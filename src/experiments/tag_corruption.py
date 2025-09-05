import os
import re
import string
import string
import pandas as pd
import random
from transformers import HfArgumentParser
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from tqdm.auto import tqdm

from utils.arguments import TagPerturbationArguments
from utils.data_utils import load_json

ACTION_REGEX = re.compile(r"\[IN:([A-Z_]+)", re.DOTALL)
SLOT_REGEX = re.compile(r"\[SL:([A-Z_]+)\s+(.*?)\]")


def choose_slot_name(span_text, pool, model):
    """
    Choose the most semantically similar slot name to the wrapped span.
    """
    slot_name_embeddings = model.encode(pool, normalize_embeddings=True)
    span_emb = model.encode([span_text], normalize_embeddings=True)
    sims = cosine_similarity(span_emb, slot_name_embeddings)[0]
    sims = list(zip(sims, pool))
    return random.choice(sorted(sims, key=lambda x: x[0], reverse=True)[:3])[1].upper()


def find_tags(text):
    """Return list of (tag_type, span_start, span_end, content)"""
    tags = []
    for m in ACTION_REGEX.finditer(text):
        tags.append(("action", m.start(), m.end(), m.group(1)))
    for m in SLOT_REGEX.finditer(text):
        tags.append(("slot", m.start(), m.end(), (m.group(1), m.group(2).strip())))
    return tags


def build_slot_value_pool(dataset):
    pool = defaultdict(set)
    for text in dataset:
        for tag in find_tags(text):
            if tag[0] == "slot":
                slot_name, slot_val = tag[3]
                pool[slot_name].add(slot_val)
    return {k: list(v) for k, v in pool.items()}


def corrupt_action(tag_val, actions):
    candidates = [a for a in actions if a != tag_val]
    return random.choice(candidates) if candidates else tag_val


def drop_slot(text, start, end):
    return text[:start] + text[start:end].split(" ", 1)[-1].strip(" ]") + text[end:]


def find_non_slot_tokens(text):
    """
    Return tokens outside [SL:...] slot spans and not starting with [IN...
    """
    slot_spans = [m.span() for m in SLOT_REGEX.finditer(text)]
    tokens = []
    for m in re.finditer(r"\S+", text):  # whitespace-split tokens
        s, e = m.span()
        inside_slot = any(s >= ss and e <= se for ss, se in slot_spans)
        token_str = text[s:e]
        if not inside_slot and not token_str.startswith(("[IN", "]")):
            tokens.append((s, e, token_str))
    return tokens


def add_spurious_slot(text, docs, model, span_len=1):
    """
    Insert a spurious slot by wrapping 1–2 tokens of non-slot text,
    and label it with the most semantically similar slot name.
    """
    tokens = find_non_slot_tokens(text)
    if not tokens:
        return text

    idx = random.randrange(len(tokens))
    end_idx = min(idx + span_len, len(tokens))

    start_char = tokens[idx][0]
    end_char = tokens[end_idx - 1][1]
    span_text = text[start_char:end_char]

    # Find the action in the utterance
    action = re.search(ACTION_REGEX, text).group(1)
    slot_pool = docs[action]

    # Pick plausible slot name
    slot_label = choose_slot_name(span_text, slot_pool, model=model)
    wrapped = f"[SL:{slot_label} {span_text} ]"

    return text[:start_char] + wrapped + text[end_char:]


def substitute_slot(slot_name, slot_val, start, end, text, docs, model):
    slot_pool = sum(
        [[i for i in x if i != slot_name.lower()] for x in docs.values()],
        [],
    )
    new_slot_name = choose_slot_name(slot_name.lower(), slot_pool, model=model)
    text = text[:start] + f"[SL:{new_slot_name} {slot_val} ]" + text[end:]
    return text


def tokenize_span(span):
    """Simple whitespace tokenizer for slot values."""
    return span.strip().split()


def can_shift_boundary(text, start, end, slot_val):
    """
    Check which boundary shifts are possible.
    Returns ['left'], ['right'], ['in'], or their combination.
    """
    options = []
    tokens = tokenize_span(slot_val)

    # Left shift check
    if start > 0:
        prev_token = text[:start].strip().split(" ")[-1]
        if prev_token not in string.punctuation and not prev_token.startswith(
            ("[IN", "]")
        ):
            options.append("left")

    # Right shift check
    if end < len(text):
        suffix = text[end:].lstrip()
        if suffix:
            next_token = suffix.split(" ", 1)[0]
            if not (
                next_token.startswith("[IN")
                or next_token.startswith("[SL:")
                or next_token[0] in string.punctuation
            ):
                options.append("right")

    # Internal shift check (only if >1 tokens in value)
    if len(tokens) > 1:
        options.append("in")

    return options


def apply_boundary_shift(text, start, end, slot_name, slot_val):
    """
    Shift slot boundary left or right (random if both possible).
    """
    options = can_shift_boundary(text, start, end, slot_val)
    if not options:
        return text

    direction = random.choice(options)

    if direction == "left":
        # include preceding token
        prefix = text[:start].rstrip()
        prev_space = prefix.rfind(" ")
        if prev_space == -1:
            prev_space = 0
        new_span = (
            text[prev_space + 1 : end]
            .replace(f"[SL:{slot_name}", "")
            .strip()
            .replace("  ", " ")
        )
        return (
            text[: prev_space + 1] + f"[SL:{slot_name} {new_span.strip()}" + text[end:]
        )

    elif direction == "right":
        # include following token
        suffix = text[end:].lstrip()
        next_token = suffix.split(" ", 1)[0]
        new_span = slot_val.strip() + " " + next_token
        after_next = text[end + len(suffix.split(" ", 1)[0]) + 1 :]
        return text[:start] + f"[SL:{slot_name} {new_span} ]" + after_next

    elif direction == "in":
        # split internal token
        tokens = tokenize_span(slot_val)
        split_idx = random.randint(1, len(tokens) - 1)
        new_span = " ".join(tokens[:split_idx])
        after_span = " ".join(tokens[split_idx:])
        return text[:start] + f"[SL:{slot_name} {new_span} ] " + after_span + text[end:]
    return text


def corrupt_dataset(dataset, error_type, n_pct, seed=13, docs=None, model=None):
    """
    dataset: list of strings (each is a tagged utterance)
    error_type: 'action' | 'slot' | 'all'
    n_pct: % of eligible tags to corrupt (global)
    """
    possible_slot_errors = [
        "slot_missing",
        "slot_boundary",
        "semantic_slot",
        # "spurious_slot",
    ]
    random.seed(seed)

    corrupted_dataset = []
    changed = []
    errors = []

    # Apply corruption

    if error_type == "spurious_slot":
        random.seed(seed)
        k = int(len(dataset) * n_pct / 100)
        chosen = set(random.sample(list(range(len(dataset))), k))
        print("TO AUGMENT:", len(chosen), "/", len(dataset))

        for ex_idx, text in tqdm(enumerate(dataset), total=len(dataset)):
            if ex_idx in chosen:
                text = add_spurious_slot(
                    text=text, docs=docs, model=model, span_len=random.choice([1, 2])
                )
            corrupted_dataset.append(text)
            changed.append(int(ex_idx in chosen))
            errors.append("spurious_slot" if ex_idx in chosen else None)

    else:
        # Collect all eligible tags
        all_tags = []
        actions = set()
        for ex_idx, text in tqdm(enumerate(dataset), total=len(dataset)):
            for tag in find_tags(text):
                if tag[0] == "action":
                    actions.add(tag[3])
                if error_type == "all" or tag[0] in error_type:
                    all_tags.append((ex_idx, tag))

        random.seed(seed)
        k = int(len(all_tags) * n_pct / 100)
        chosen = set(random.sample(all_tags, k))

        print("TO AUGMENT:", len(chosen), "/", len(all_tags))

        if error_type == "all":
            for ex_idx, text in enumerate(dataset):
                has_changes = False
                instance_errors = []
                tags = find_tags(text)
                for tag in reversed(tags):  # right-to-left so indices stay valid
                    tag_type, start, end, content = tag

                    if (ex_idx, tag) not in chosen:
                        continue
                    else:
                        has_changes = True

                    if tag_type == "action":
                        new_action = corrupt_action(tag_val=content, actions=actions)
                        text = (
                            text[:start]
                            + f"[IN:{new_action}"
                            + text[start + len(f"[IN:{content}") :]
                        )
                        instance_errors.append("action")

                    else:
                        error = random.choice(possible_slot_errors)
                        instance_errors.append(error)

                        if error == "slot_missing":
                            text = drop_slot(text=text, start=start, end=end)
                        elif error == "semantic_slot":
                            text = substitute_slot(
                                slot_name=content[0],
                                slot_val=content[1],
                                start=start,
                                end=end,
                                text=text,
                                docs=docs,
                                model=model,
                            )
                        elif error == "slot_boundary":
                            text = apply_boundary_shift(
                                text=text,
                                start=start,
                                end=end,
                                slot_name=content[0],
                                slot_val=content[1],
                            )
                        # elif error == "spurious_slot":
                        #     text = add_spurious_slot(text, span_len=1)

                corrupted_dataset.append(text)
                changed.append(int(has_changes))
                errors.append(", ".join(instance_errors))

        else:
            for ex_idx, text in enumerate(dataset):
                has_changes = False
                tags = find_tags(text)
                for tag in reversed(tags):  # right-to-left so indices stay valid
                    tag_type, start, end, content = tag

                    if (ex_idx, tag) not in chosen:
                        continue
                    else:
                        has_changes = True

                    if tag_type == "action" and error_type == "action":
                        new_action = corrupt_action(tag_val=content, actions=actions)
                        text = (
                            text[:start]
                            + f"[IN:{new_action}"
                            + text[start + len(f"[IN:{content}") :]
                        )

                    elif tag_type == "slot":
                        if error_type == "slot_missing":
                            text = drop_slot(text=text, start=start, end=end)
                        elif error_type == "semantic_slot":
                            text = substitute_slot(
                                slot_name=content[0],
                                slot_val=content[1],
                                start=start,
                                end=end,
                                text=text,
                                docs=docs,
                                model=model,
                            )
                        elif error_type == "slot_boundary":
                            text = apply_boundary_shift(
                                text=text,
                                start=start,
                                end=end,
                                slot_name=content[0],
                                slot_val=content[1],
                            )

                corrupted_dataset.append(text)
                changed.append(int(has_changes))
                errors.append(error_type if has_changes else None)

    corrupted_dataset = pd.DataFrame(
        {
            "org_aug": dataset,
            "aug": corrupted_dataset,
            "corrupted": changed,
            "errors": errors,
        }
    )
    return corrupted_dataset


def main(args):

    docs, model = None, None
    if args.strategy in ("all", "semantic_slot", "spurious_slot"):
        model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
        docs = load_json(args.api_schema_path)
    

    tags_dataset = pd.read_csv(args.input_fpath)
    if args.max_samples:
        tags_dataset = tags_dataset.iloc[: args.max_samples]

    print(f"Loaded {len(tags_dataset)} examples from {args.input_fpath}")
    corrupted_dataset = corrupt_dataset(
        dataset=tags_dataset["aug"].tolist(),
        error_type=args.strategy,
        n_pct=args.n_pct,
        seed=args.seed,
        docs=docs,
        model=model,
    )

    corrupted_dataset['org'] = tags_dataset['org']

    dataset_name = args.input_fpath.split("/")[-1].replace(".csv", "")
    output_dir = os.path.join(args.output_dir, args.strategy)
    output_fpath = os.path.join(
        output_dir,
        f"corrupted__{dataset_name}__{args.strategy}__{args.n_pct}__seed_{args.seed}.csv",
    )

    print(f"Writing to {output_fpath}...")
    os.makedirs(output_dir, exist_ok=True)
    corrupted_dataset.to_csv(output_fpath, index=False)


if __name__ == "__main__":
    parser = HfArgumentParser(TagPerturbationArguments)
    args = parser.parse_args_into_dataclasses()[0]
    main(args)
