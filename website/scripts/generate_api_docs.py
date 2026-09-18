"""
Generate API documentation for the Opytimizer package.

This script discovers public classes inside the ``opytimizer`` package and
generates Markdown files that can be consumed directly by Docusaurus.

The generated documentation contains:

- Class name and module;
- Class docstring;
- Constructor signature;
- Constructor parameters;
- Public methods;
- Method signatures;
- Method parameters;
- Method docstrings.

Generated files are written to:

    docs/api/

The generated directory can safely be removed and regenerated at any time.

Usage:

    python scripts/generate_api_docs.py
"""

from __future__ import annotations

import importlib
import inspect
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator, Optional


# ==============================================================================
# Configuration
# ==============================================================================

PACKAGE_NAME = "opytimizer"

SCRIPT_PATH = Path(__file__).resolve()

PROJECT_ROOT = SCRIPT_PATH.parents[2]
WEBSITE_DIR = SCRIPT_PATH.parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

PACKAGE_DIR = PROJECT_ROOT / PACKAGE_NAME

DOCS_DIR = WEBSITE_DIR / "docs"
API_DIR = DOCS_DIR / "api"


# Modules that should never be documented.
IGNORED_MODULES = {
    "__init__",
}

# Classes whose names start with "_" are considered private.
INCLUDE_PRIVATE_CLASSES = False

# Methods whose names start with "_" are considered private.
INCLUDE_PRIVATE_METHODS = False


# ==============================================================================
# Data structures
# ==============================================================================


@dataclass
class Parameter:
    """Represents a documented function parameter."""

    name: str
    annotation: str
    default: str
    description: str


@dataclass
class MethodDocumentation:
    """Represents the documentation of a method."""

    name: str
    signature: str
    description: str
    parameters: list[Parameter]


# ==============================================================================
# General helpers
# ==============================================================================


def clean_docstring(obj: Any) -> str:
    """
    Return a cleaned docstring.

    Parameters
    ----------
    obj
        Python object whose docstring should be extracted.

    Returns
    -------
    str
        Cleaned docstring or an empty string if no docstring exists.
    """

    doc = inspect.getdoc(obj)

    if not doc:
        return ""

    return doc.strip()


def format_annotation(annotation: Any) -> str:
    """
    Format a Python type annotation for Markdown.

    Parameters
    ----------
    annotation
        Annotation returned by ``inspect.signature``.

    Returns
    -------
    str
        Human-readable annotation.
    """

    if annotation is inspect.Parameter.empty:
        return ""

    if isinstance(annotation, str):
        return annotation

    # Python classes and typing objects.
    try:
        text = inspect.formatannotation(annotation)
    except (TypeError, ValueError):
        text = str(annotation)

    # Remove the module prefix from common Python types.
    text = text.replace("typing.", "")
    text = text.replace("builtins.", "")

    return text


def format_default(default: Any) -> str:
    """
    Format a parameter default value.

    Parameters
    ----------
    default
        Default value returned by ``inspect.signature``.

    Returns
    -------
    str
        Markdown-safe representation of the default value.
    """

    if default is inspect.Parameter.empty:
        return ""

    return repr(default)


def escape_markdown(text: str) -> str:
    """
    Escape characters that can interfere with Markdown tables.

    Parameters
    ----------
    text
        Text to escape.

    Returns
    -------
    str
        Escaped text.
    """

    return text.replace("|", "\\|").replace("\n", " ").strip()


def module_to_category(module_name: str) -> str:
    """
    Determine the API documentation category for a module.

    Examples
    --------
    ``opytimizer.optimizers.population`` becomes ``optimizers``.

    ``opytimizer.spaces.graph`` becomes ``spaces``.
    """

    parts = module_name.split(".")

    if len(parts) < 2:
        return "other"

    return parts[1]


def module_to_output_path(module_name: str, class_name: str) -> Path:
    """
    Determine the output path for a class documentation page.

    The module hierarchy is preserved below ``docs/api``.

    For example:

        opytimizer.optimizers.single_objective.swarm.aco.ACO

    becomes:

        docs/api/optimizers/single_objective/swarm/aco.md
    """

    parts = module_name.split(".")

    if parts[0] == PACKAGE_NAME:
        parts = parts[1:]

    module_path = Path(*parts)

    filename = f"{class_name.lower()}.md"

    return API_DIR / module_path.parent / filename


# ==============================================================================
# Module discovery
# ==============================================================================


def iter_python_modules(package_dir: Path) -> Iterator[str]:
    """
    Discover all Python modules inside the package directory.

    Parameters
    ----------
    package_dir
        Root directory of the Python package.

    Yields
    ------
    str
        Fully qualified module names.
    """

    package_prefix = PACKAGE_NAME

    for path in sorted(package_dir.rglob("*.py")):
        relative = path.relative_to(package_dir)

        # Ignore Python cache directories.
        if "__pycache__" in relative.parts:
            continue

        # Ignore private modules.
        if any(part.startswith("_") for part in relative.parts):
            continue

        parts = list(relative.parts)

        if parts[-1] == "__init__.py":
            parts = parts[:-1]

            if not parts:
                module_name = package_prefix
            else:
                module_name = ".".join([package_prefix, *parts])
        else:
            parts[-1] = parts[-1][:-3]
            module_name = ".".join([package_prefix, *parts])

        if module_name in IGNORED_MODULES:
            continue

        yield module_name


def import_module(module_name: str) -> Optional[ModuleType]:
    """
    Import a module safely.

    Parameters
    ----------
    module_name
        Fully qualified module name.

    Returns
    -------
    ModuleType or None
        Imported module, or ``None`` if the import failed.
    """

    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        print(
            f"WARNING: Could not import {module_name}: {exc}",
            file=sys.stderr,
        )
        return None


# ==============================================================================
# Class discovery
# ==============================================================================


def is_public_class(cls: type) -> bool:
    """
    Determine whether a class should be included in the API documentation.
    """

    if not INCLUDE_PRIVATE_CLASSES and cls.__name__.startswith("_"):
        return False

    return True


def is_class_defined_in_module(cls: type, module: ModuleType) -> bool:
    """
    Check whether a class is actually defined by a module.

    This prevents imported classes from appearing in multiple API pages.
    """

    return cls.__module__ == module.__name__


def discover_classes(module: ModuleType) -> list[type]:
    """
    Discover public classes defined by a module.

    Parameters
    ----------
    module
        Python module to inspect.

    Returns
    -------
    list[type]
        Classes defined in the module.
    """

    classes = []

    for _, cls in inspect.getmembers(module, inspect.isclass):
        if not is_public_class(cls):
            continue

        if not is_class_defined_in_module(cls, module):
            continue

        classes.append(cls)

    return sorted(classes, key=lambda cls: cls.__name__)


# ==============================================================================
# Docstring parsing
# ==============================================================================


def normalize_docstring(doc: str) -> str:
    """
    Normalize common Python docstring formatting.

    This function intentionally performs only lightweight processing.
    """

    lines = doc.splitlines()

    normalized = []

    for line in lines:
        normalized.append(line.rstrip())

    return "\n".join(normalized).strip()


def find_parameter_section(
    doc: str,
) -> tuple[list[str], list[str]]:
    """
    Extract parameter descriptions from a docstring.

    Supported section names include:

    - Args
    - Arguments
    - Parameters
    - Keyword Args
    - Keyword Arguments

    Returns
    -------
    tuple
        Parameter lines and remaining documentation lines.
    """

    if not doc:
        return [], []

    lines = doc.splitlines()

    section_names = {
        "args",
        "arguments",
        "parameters",
        "keyword args",
        "keyword arguments",
    }

    parameter_lines: list[str] = []
    remaining_lines: list[str] = []

    in_parameter_section = False
    section_indent: Optional[int] = None

    for line in lines:
        stripped = line.strip()

        # Detect a section heading.
        if stripped.rstrip(":").lower() in section_names:
            in_parameter_section = True
            section_indent = None
            continue

        if in_parameter_section:
            if not stripped:
                continue

            indent = len(line) - len(line.lstrip())

            if section_indent is None:
                section_indent = indent

            # A new unindented section means the parameter section ended.
            if indent < section_indent:
                in_parameter_section = False
                remaining_lines.append(line)
                continue

            parameter_lines.append(line)
        else:
            remaining_lines.append(line)

    return parameter_lines, remaining_lines


def parse_parameter_descriptions(
    doc: str,
) -> dict[str, str]:
    """
    Parse parameter descriptions from a docstring.

    Supports common Google-style syntax such as:

        Args:
            alpha: Pheromone importance.
            beta: Heuristic importance.

    and:

        Parameters:
            alpha : float
                Pheromone importance.
    """

    if not doc:
        return {}

    lines = doc.splitlines()

    section_names = {
        "args",
        "arguments",
        "parameters",
        "keyword args",
        "keyword arguments",
    }

    descriptions: dict[str, str] = {}

    in_section = False
    current_name: Optional[str] = None

    for line in lines:
        stripped = line.strip()

        if stripped.rstrip(":").lower() in section_names:
            in_section = True
            current_name = None
            continue

        if not in_section:
            continue

        # Detect another section heading.
        if (
            stripped
            and not line.startswith((" ", "\t"))
            and stripped.endswith(":")
        ):
            in_section = False
            current_name = None
            continue

        if not stripped:
            continue

        # Google style:
        #
        #     alpha: Pheromone importance.
        #
        # Also supports:
        #
        #     alpha (float): Pheromone importance.
        #
        #     alpha: Pheromone importance.
        match = re.match(
            r"^\s*([*\w][\w,\s\*\-]*)\s*(?:\([^)]*\))?\s*:\s*(.*)$",
            line,
        )

        if match:
            names = [
                name.strip().lstrip("*")
                for name in match.group(1).split(",")
            ]

            description = match.group(2).strip()

            for name in names:
                descriptions[name] = description

            current_name = names[0]
            continue

        # NumPy-style:
        #
        #     alpha : float
        #
        match = re.match(
            r"^\s*([*\w][\w\*\-]*)\s*:\s*(.*)$",
            line,
        )

        if match:
            current_name = match.group(1).lstrip("*")
            descriptions[current_name] = ""
            continue

        # Continuation line.
        if current_name is not None:
            descriptions[current_name] = (
                f"{descriptions[current_name]} {stripped}"
            ).strip()

    return descriptions


# ==============================================================================
# Signature handling
# ==============================================================================


def get_signature(obj: Any) -> str:
    """
    Safely obtain a callable signature.
    """

    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "()"


def get_parameters(
    obj: Any,
    descriptions: Optional[dict[str, str]] = None,
) -> list[Parameter]:
    """
    Extract parameters from a callable.

    Parameters
    ----------
    obj
        Callable object.
    descriptions
        Optional parameter descriptions extracted from its docstring.
    """

    descriptions = descriptions or {}

    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):
        return []

    parameters = []

    for parameter in signature.parameters.values():
        if parameter.name in {"self", "cls"}:
            continue

        annotation = format_annotation(parameter.annotation)
        default = format_default(parameter.default)

        description = descriptions.get(parameter.name, "")

        parameters.append(
            Parameter(
                name=parameter.name,
                annotation=annotation,
                default=default,
                description=description,
            )
        )

    return parameters


# ==============================================================================
# Markdown generation
# ==============================================================================


def generate_parameter_table(
    parameters: list[Parameter],
) -> list[str]:
    """
    Generate a Markdown parameter table.
    """

    if not parameters:
        return []

    lines = [
        "### Parameters",
        "",
        "| Parameter | Type | Default | Description |",
        "| --- | --- | --- | --- |",
    ]

    for parameter in parameters:
        name = f"`{parameter.name}`"

        annotation = (
            f"`{parameter.annotation}`"
            if parameter.annotation
            else ""
        )

        default = (
            f"`{parameter.default}`"
            if parameter.default
            else ""
        )

        description = escape_markdown(
            parameter.description or "—"
        )

        lines.append(
            f"| {name} | {annotation} | {default} | {description} |"
        )

    lines.append("")

    return lines


def generate_method_documentation(
    method: Any,
    name: str,
) -> MethodDocumentation:
    """
    Generate documentation metadata for a method.
    """

    doc = normalize_docstring(clean_docstring(method))

    parameter_descriptions = parse_parameter_descriptions(doc)

    parameters = get_parameters(
        method,
        parameter_descriptions,
    )

    # Remove parameter sections from the main description.
    _, description_lines = find_parameter_section(doc)

    description = "\n".join(description_lines).strip()

    return MethodDocumentation(
        name=name,
        signature=get_signature(method),
        description=description,
        parameters=parameters,
    )


def discover_public_methods(
    cls: type,
) -> list[tuple[str, Any]]:
    """
    Discover public methods defined directly by a class.
    """

    methods = []

    for name, method in inspect.getmembers(cls, inspect.isfunction):
        if not INCLUDE_PRIVATE_METHODS and name.startswith("_"):
            continue

        # Only document methods defined directly by this class.
        if method.__qualname__.split(".")[0] != cls.__name__:
            continue

        methods.append((name, method))

    return sorted(methods, key=lambda item: item[0])


def generate_class_markdown(
    cls: type,
) -> str:
    """
    Generate complete Markdown documentation for a class.
    """

    doc = normalize_docstring(clean_docstring(cls))

    parameter_descriptions = parse_parameter_descriptions(doc)

    constructor_parameters = get_parameters(
        cls,
        parameter_descriptions,
    )

    _, class_description_lines = find_parameter_section(doc)

    class_description = "\n".join(
        class_description_lines
    ).strip()

    module_name = cls.__module__

    lines: list[str] = []

    # ==============================================================================
    # Front matter
    # ==============================================================================

    lines.extend(
        [
            "---",
            f"title: {cls.__name__}",
            f"description: API reference for {cls.__name__}.",
            "---",
            "",
        ]
    )

    # ==============================================================================
    # Class
   # ==============================================================================

    lines.extend(
        [
            f"# `{cls.__name__}`",
            "",
            f"**Module:** `{module_name}`",
            "",
        ]
    )

    if class_description:
        lines.extend(
            [
                class_description,
                "",
            ]
        )

    # ==============================================================================
    # Constructor
    # ==============================================================================

    lines.extend(
        [
            "## Constructor",
            "",
            "```python",
            f"{cls.__name__}{get_signature(cls)}",
            "```",
            "",
        ]
    )

    lines.extend(
        generate_parameter_table(
            constructor_parameters
        )
    )

    # ==============================================================================
    # Methods
    # ==============================================================================

    methods = discover_public_methods(cls)

    if methods:
        lines.extend(
            [
                "## Methods",
                "",
            ]
        )

    for name, method in methods:
        documentation = generate_method_documentation(
            method,
            name,
        )

        lines.extend(
            [
                f"### `{documentation.name}`",
                "",
                "```python",
                (
                    f"{documentation.name}"
                    f"{documentation.signature}"
                ),
                "```",
                "",
            ]
        )

        if documentation.description:
            lines.extend(
                [
                    documentation.description,
                    "",
                ]
            )

        lines.extend(
            generate_parameter_table(
                documentation.parameters
            )
        )

    return "\n".join(lines).rstrip() + "\n"



# ==============================================================================
# File generation
# ==============================================================================



def prepare_output_directory() -> None:
    """
    Prepare the generated API documentation directory.

    The complete ``docs/api`` directory is removed before generation.
    This prevents stale documentation from remaining after classes/modules
    are deleted from the Python package.
    """

    if API_DIR.exists():
        shutil.rmtree(API_DIR)

    API_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def write_class_documentation(cls: type) -> Path:
    """
    Write a class documentation page.
    """

    output_file = module_to_output_path(
        cls.__module__,
        cls.__name__,
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    content = generate_class_markdown(cls)

    output_file.write_text(
        content,
        encoding="utf-8",
    )

    return output_file


# ==============================================================================
# Main generation proccess
# ==============================================================================


def generate_api_documentation() -> int:
    """
    Generate API documentation for the complete package.

    Returns
    -------
    int
        Number of generated documentation pages.
    """

    if not PACKAGE_DIR.exists():
        raise FileNotFoundError(
            f"Could not find package directory: {PACKAGE_DIR}"
        )

    print(
        f"Generating API documentation for "
        f"{PACKAGE_NAME}..."
    )

    prepare_output_directory()

    generated_files = 0

    modules = list(
        iter_python_modules(PACKAGE_DIR)
    )

    print(
        f"Discovered {len(modules)} Python modules."
    )

    for module_name in modules:
        print(f"Inspecting {module_name}")

        module = import_module(module_name)

        if module is None:
            continue

        classes = discover_classes(module)

        for cls in classes:
            output_file = write_class_documentation(
                cls
            )

            print(
                f"  Generated: {output_file.relative_to(PROJECT_ROOT)}"
            )

            generated_files += 1

    print()
    print(
        f"Generated {generated_files} API documentation pages."
    )

    return generated_files


def main() -> None:
    """Main entry point."""

    try:
        generate_api_documentation()
    except Exception as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        raise


if __name__ == "__main__":
    main()