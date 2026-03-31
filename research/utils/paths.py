from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent.resolve()


def get_mirror_path(current_file: str, target_module: str) -> Path:
    abs_current = Path(current_file).resolve()
    root = get_project_root()

    try:
        rel_path = abs_current.relative_to(root)
        sub_structure = Path(*rel_path.parts[1:-1])
        target_path = root / target_module / sub_structure
        target_path.mkdir(parents=True, exist_ok=True)
        return target_path
    except ValueError:
        return abs_current.parent
