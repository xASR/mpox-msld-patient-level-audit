"""Export hash-verified modules in the layout expected by a study workflow."""

import argparse
import csv
from hashlib import sha256
from pathlib import Path, PurePosixPath


def load_sources(repository: Path, workflow: str) -> dict[str, bytes]:
    index = repository / "reproducibility" / "code_sources.csv"
    modules = {}
    with index.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["workflow"] != workflow:
                continue
            name = row["runtime_name"]
            source = PurePosixPath(row["repository_path"])
            if (
                source.is_absolute()
                or ".." in source.parts
                or not source.parts
                or source.parts[0] != "src"
                or Path(name).name != name
                or not name.endswith(".py")
                or name in modules
            ):
                raise ValueError(f"Invalid source path for {workflow}: {name}")
            local_path = repository.joinpath(*source.parts).resolve()
            if not local_path.is_relative_to(repository.resolve()):
                raise ValueError(f"Source leaves the repository: {local_path}")
            data = local_path.read_bytes()
            if sha256(data).hexdigest() != row["sha256"]:
                raise ValueError(f"Code checksum differs: {source}")
            modules[name] = data
    if not modules:
        raise ValueError(f"Unknown workflow: {workflow}")
    return modules


def export_workflow(repository: Path, workflow: str, destination: Path) -> None:
    modules = load_sources(repository, workflow)
    for name, data in modules.items():
        target = destination / name
        if target.is_symlink() or (target.exists() and target.read_bytes() != data):
            raise ValueError(f"Choose an empty output directory: {target}")
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in modules.items():
        target = destination / name
        if not target.exists():
            with target.open("xb") as handle:
                handle.write(data)
    print(f"Verified {len(modules)} modules for {workflow}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow")
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export_workflow(Path(__file__).resolve().parents[1], args.workflow, args.destination)


if __name__ == "__main__":
    main()
