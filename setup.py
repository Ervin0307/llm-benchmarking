import os
from setuptools import find_packages, setup
from typing import Dict, List

ROOT_DIR = os.path.dirname(__file__)

def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)

def _read_requirements(filename: str) -> List[str]:
    with open(get_path(filename)) as f:
        requirements = f.read().strip().split("\n")
    resolved_requirements = []
    for line in requirements:
        if line.startswith("-r "):
            resolved_requirements += _read_requirements(line.split()[1])
        elif line.startswith("--"):
            continue
        else:
            resolved_requirements.append(line)
    print(resolved_requirements)
    return resolved_requirements

setup(
    author="Bud Ecosystem",
    python_requires=">=3.10",
    description="A LLM inference benchmarking tool",
    include_package_data=True,
    keywords="bud",
    name="llm_benchmark",
    packages=find_packages(include=["llm_benchmark", "llm_benchmark.*"]),
    install_requires=_read_requirements('requirements.txt'),
    version="0.0.2",
)
