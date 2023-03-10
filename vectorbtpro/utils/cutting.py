# Copyright (c) 2023 Oleg Polakow. All rights reserved.

"""Utilities for cutting code."""

import inspect
from types import ModuleType, FunctionType
import importlib
from pathlib import Path

from vectorbtpro import _typing as tp
from vectorbtpro.utils.template import CustomTemplate, RepEval

__all__ = [
    "cut_and_save_module",
    "cut_and_save_func",
]


def cut_from_code(
    code: str,
    section_name: str,
    addons: tp.Optional[tp.Iterable[str]] = None,
    prepend_lines: tp.Optional[tp.Iterable[str]] = None,
    append_lines: tp.Optional[tp.Iterable[str]] = None,
    new_lines_callback: tp.Union[None, tp.Callable, CustomTemplate] = None,
    return_lines: bool = False,
    **kwargs,
) -> tp.Union[str, tp.List[str]]:
    """Cut an annotated section from the code.

    The respective section in the code should start with `# <section_name>` and end with `# </>`.

    Within the section, any line that starts with a comment will be uncommented.

    If the line ends with a comment that starts with a percentage sign (%), then everything
    after the sign will be treated like a command. The only built-in command is `skip` to skip the line.

    If the percentage sign is followed by a question mark (?), then the command will be evaluated
    using `vectorbtpro.utils.template.RepEval` with the context including `section_name`, `line`, `line_nr`,
    and unpacked `kwargs`. If the template evaluation fails, the original line will be preserved.
    If the percentage sign is followed by an exclamation mark (!) instead, an error will be thrown."""
    lines = code.split("\n")
    if addons is None:
        addons = []
    else:
        if isinstance(addons, str):
            addons = [addons]
        else:
            addons = list(addons)
    new_lines = []
    section_found = False
    uncomment = False
    skip = False
    no_addon_skip = False
    addon_start_line_nr = None
    addon_index = None
    line_nr = 1

    while line_nr <= len(lines):
        line = lines[line_nr - 1]
        sline = line.strip()

        if sline.startswith("# % <section:") and sline.endswith(">"):
            section_found = sline[len("# % <section:"):-1].strip() == section_name
        elif section_found:
            context = {
                "section_name": section_name,
                "addons": addons,
                "addon_name": None if addon_index is None else addons[addon_index],
                "line": line,
                "line_nr": line_nr,
                "new_lines": new_lines,
                **kwargs,
            }
            if no_addon_skip:
                if sline.startswith("# % <for-each-addon>"):
                    no_addon_skip = False
            else:
                if sline.startswith("# % </skip>"):
                    skip = False
                elif not skip:
                    if sline.startswith("# % <skip>"):
                        skip = True
                    elif sline.startswith("# % <skip:") and sline.endswith(">"):
                        expression = sline[len("# % <skip:"):-1].strip()
                        if RepEval(expression).substitute(context=context):
                            skip = True
                    elif sline.startswith("# % <for-each-addon>"):
                        if len(addons) == 0:
                            no_addon_skip = True
                        else:
                            addon_index = 0
                            addon_start_line_nr = line_nr
                    elif sline.startswith("# % insert_addon"):
                        addon_lines = cut_from_code(code, addons[addon_index], return_lines=True, **kwargs)
                        new_lines.extend(addon_lines)
                    elif sline.startswith("# % </for-each-addon>"):
                        addon_index += 1
                        if addon_index < len(addons):
                            line_nr = addon_start_line_nr + 1
                            continue
                        else:
                            addon_index = None
                            addon_start_line_nr = None
                    elif uncomment:
                        if sline.startswith("# % </uncomment>"):
                            uncomment = False
                        elif sline.startswith("# "):
                            new_lines.append(sline[2:])
                        elif sline.startswith("#"):
                            new_lines.append(sline[1:])
                    else:
                        if sline.startswith("# % </section>"):
                            if prepend_lines is not None:
                                new_lines.extend(list(prepend_lines))
                            if append_lines is not None:
                                new_lines.extend(list(append_lines))
                            if new_lines_callback is not None:
                                if isinstance(new_lines_callback, CustomTemplate):
                                    new_lines_callback = new_lines_callback.substitute(context=context)
                                new_lines = new_lines_callback(new_lines)
                            if return_lines:
                                return new_lines
                            return inspect.cleandoc("\n".join(new_lines))
                        if sline.startswith("# % <uncomment>"):
                            uncomment = True
                        elif sline.startswith("# % <uncomment ?") and sline.endswith(">"):
                            flag_command = sline[len("# % <uncomment ?"):-1].strip()
                            if "(" in flag_command and flag_command.endswith(")"):
                                flag_name = flag_command.split("(")[0].strip()
                                if flag_name in kwargs:
                                    flag = kwargs[flag_name]
                                else:
                                    flag = eval(flag_command.split("(")[1][:-1].strip(), {})
                            else:
                                flag = kwargs[flag_command]
                            if flag:
                                uncomment = True
                        elif "# %?" in line or "# %!" in line:
                            if "# %?" in line:
                                sep = "# %?"
                                strict = False
                            else:
                                sep = "# %!"
                                strict = True
                            expression = line.split(sep)[1].strip()
                            line_woc = line.split(sep)[0].rstrip()
                            context["line"] = line_woc
                            eval_line = RepEval(expression).substitute(context=context, strict=strict)
                            if not isinstance(eval_line, RepEval):
                                line = eval_line
                            new_lines.append(line)
                        elif "# %" in line:
                            command = line.split("# %")[1].strip()
                            if command.lower() == "skip":
                                line_nr += 1
                                continue
                            if command.startswith("replace(") and command.endswith(")"):
                                new_lines.append(line)
                            else:
                                raise ValueError(f"Invalid command '{command}'")
                        else:
                            new_lines.append(line)
        line_nr += 1
    raise ValueError(f"Code section '{section_name}' not found")


def cut_and_save(code: str, section_name: str, path: tp.Optional[tp.PathLike] = None, **kwargs) -> Path:
    """Cut an annotated section from the code and save to a file.

    For arguments see `cut_from_code`."""
    parsed_code = cut_from_code(code, section_name, **kwargs)
    if path is None:
        path = Path(".")
    else:
        path = Path(path)
    if path.is_dir():
        path = (path / section_name).with_suffix(".py")
    with open(path, "w") as f:
        f.write(parsed_code)
    return path


def cut_and_save_module(module: tp.Union[str, ModuleType], *args, **kwargs) -> Path:
    """Cut an annotated section from a module and save to a file.

    For arguments see `cut_and_save`."""
    if isinstance(module, str):
        module = importlib.import_module(module)
    code = inspect.getsource(module)
    return cut_and_save(code, *args, **kwargs)


def cut_and_save_func(func: tp.Union[str, FunctionType], *args, **kwargs) -> Path:
    """Cut an annotated function section from a module and save to a file.

    For arguments see `cut_and_save`."""
    if isinstance(func, str):
        module = importlib.import_module(".".join(func.split(".")[:-1]))
        func = getattr(module, func.split(".")[-1])
    else:
        module = inspect.getmodule(func)
    code = inspect.getsource(module)
    return cut_and_save(code, section_name=func.__name__, *args, **kwargs)
