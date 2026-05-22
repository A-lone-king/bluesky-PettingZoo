"""Script to replace duplicated test helpers with shared imports."""

import re
import pathlib

def remove_block(content: str, start_pattern: str, end_patterns: list[str]) -> tuple[str, bool]:
    """Remove a block of code from start_pattern to the next end_pattern."""
    match = re.search(start_pattern, content)
    if not match:
        return content, False

    start = match.start()
    # Find the end - look for the next class, def at same level, or section separator
    rest = content[match.end():]
    end = -1
    for pat in end_patterns:
        m = re.search(pat, rest)
        if m and (end == -1 or m.start() < end):
            end = m.start()

    if end == -1:
        # Block goes to end of file
        return content[:start].rstrip() + "\n", True

    return content[:start] + rest[end:], True


def process_file(filepath: str) -> None:
    p = pathlib.Path(filepath)
    if not p.exists():
        print(f"SKIP: {filepath}")
        return

    content = p.read_text(encoding="utf-8")
    original = content
    changed = False

    # Remove FakeBlueSkyWrapper class
    content, removed = remove_block(
        content,
        r"class FakeBlueSkyWrapper:",
        [r"\n# -{10,}\n", r"\nclass [A-Z]", r"\ndef [a-z_]"],
    )
    if removed:
        changed = True
        print(f"  Removed FakeBlueSkyWrapper from {filepath}")

    # Remove _make_config function
    content, removed = remove_block(
        content,
        r"def _make_config\(",
        [r"\n# -{10,}\n", r"\ndef [a-z_]", r"\nclass [A-Z]"],
    )
    if removed:
        changed = True
        print(f"  Removed _make_config from {filepath}")

    # Remove _write_rewards_yaml function
    content, removed = remove_block(
        content,
        r"def _write_rewards_yaml\(",
        [r"\n# -{10,}\n", r"\ndef [a-z_]", r"\nclass [A-Z]"],
    )
    if removed:
        changed = True
        print(f"  Removed _write_rewards_yaml from {filepath}")

    # Remove _make_rewards_config function
    content, removed = remove_block(
        content,
        r"def _make_rewards_config\(",
        [r"\n# -{10,}\n", r"\ndef [a-z_]", r"\nclass [A-Z]"],
    )
    if removed:
        changed = True
        print(f"  Removed _make_rewards_config from {filepath}")

    # Remove _make_env function
    content, removed = remove_block(
        content,
        r"def _make_env\(",
        [r"\n# -{10,}\n", r"\ndef [a-z_]", r"\nclass [A-Z]"],
    )
    if removed:
        changed = True
        print(f"  Removed _make_env from {filepath}")

    if changed:
        # Add import for helpers if not already present
        if "from tests.helpers" not in content:
            # Find the last import line
            lines = content.split("\n")
            last_import = 0
            for i, line in enumerate(lines):
                if line.startswith("from ") or line.startswith("import "):
                    last_import = i

            # Determine what imports are needed
            imports = []
            if "FakeBlueSkyWrapper" in original and "class FakeBlueSkyWrapper" not in content:
                imports.append("from tests.helpers.fake_wrapper import FakeBlueSkyWrapper")
            if "_make_config" in original and "def _make_config" not in content:
                imports.append("from tests.helpers.env_factory import make_config as _make_config")
            if "_write_rewards_yaml" in original and "def _write_rewards_yaml" not in content:
                imports.append("from tests.helpers.env_factory import write_rewards_yaml as _write_rewards_yaml")
            if "_make_rewards_config" in original and "def _make_rewards_config" not in content:
                imports.append("from tests.helpers.env_factory import _DEFAULT_REWARDS as _make_rewards_config")
            if "_make_env" in original and "def _make_env" not in content:
                imports.append("from tests.helpers.env_factory import make_env as _make_env")

            if imports:
                lines.insert(last_import + 1, "\n" + "\n".join(imports))
                content = "\n".join(lines)

        p.write_text(content, encoding="utf-8")
        print(f"  Updated {filepath}")
    else:
        print(f"  No changes: {filepath}")


files = [
    "tests/integration/test_performance.py",
    "tests/integration/test_scenario_e2e.py",
    "tests/integration/test_backward_compat.py",
    "tests/integration/test_merge.py",
    "tests/integration/test_waypoint_nav.py",
    "tests/integration/test_sector_cr.py",
    "tests/integration/test_vertical_cr.py",
    "tests/integration/test_horizontal_cr.py",
    "tests/test_env.py",
    "tests/test_arrival_termination.py",
    "tests/test_dynamic_entry.py",
    "tests/test_api_compliance.py",
    "tests/integration/test_no_conflict.py",
    "tests/integration/test_single_conflict.py",
    "tests/integration/test_multi_conflict.py",
    "tests/test_action_frequency.py",
]

for f in files:
    print(f"\nProcessing {f}...")
    process_file(f)
