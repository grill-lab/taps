import re
import numpy as np


def match_args(func: str, skip_to: int = -1, debug: bool = False) -> tuple:
    if debug:
        print("START")
        print(f"skip_to={skip_to}")

    args = []
    tokens = func.split()
    cur_func_args = []
    cur_func = None
    for t, token in enumerate(tokens):
        if debug:
            print(t, token)
        if t < skip_to:
            if debug:
                print(f"{t}: skipping to {skip_to}")
            continue
        if token.startswith("[IN:"):
            if cur_func is not None:
                other_args, skip_to = match_args(func, skip_to=t)
                args.extend(other_args)
            else:
                cur_func = token.split(":")[1]
        elif token.startswith("]"):
            if debug:
                print("FINISH")
            skip_to = t
            args.append({"func": cur_func, "args": cur_func_args})
            return args, skip_to
        elif token.startswith("arg_"):
            cur_func_args.append(token)
            if debug:
                print(f"added to {cur_func}")
    if debug:
        print()

    return args, skip_to


def parse2python(tagged_input: str) -> list:
    args = re.findall("\[SL:.+?\]", tagged_input)
    formatted_args = []
    for arg in args:
        arg_name = arg.split(":")[1].split()[0].lower()
        value = arg.split(" ", 1)
        if len(value) > 1:
            value = value[1].split("]")[0].strip().strip('"')
        else:
            value = ""
        form_arg = f'{arg_name}="{value}"'
        formatted_args.append(form_arg)
    formatted_args = np.array(formatted_args)

    for i, arg in enumerate(args):
        tagged_input = tagged_input.replace(arg, f"arg_{i}")

    funcs = re.findall("(?:\[IN:(.+?) )", tagged_input)
    output = []
    if len(funcs) == 1:
        func_name = "".join([i[0].title() + i[1:] for i in funcs[0].lower().split("_")])
        output.append(f'{func_name}({", ".join(formatted_args)})')
        return output

    parsed_funcs, _ = match_args(tagged_input)
    for func in parsed_funcs:
        if func["func"] is None:
            continue

        func_name = "".join(
            [i[0].title() + i[1:] for i in func["func"].lower().split("_")]
        )
        arg_ids = list(map(lambda x: int(x.split("_")[-1].strip(",.")), func["args"]))
        func_args = ", ".join(formatted_args[arg_ids].tolist())
        output.append(f"{func_name}({func_args})")

    return output
