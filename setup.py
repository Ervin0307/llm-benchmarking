from setuptools import find_packages, setup


setup(
    author="Bud Ecosystem",
    python_requires=">=3.10",
    description="A LLM inference benchmarking tool",
    include_package_data=True,
    keywords="bud",
    name="llm_benchmark",
    packages=find_packages(include=["llm_benchmark", "llm_benchmark.*"]),
    install_requires=open("requirements.txt").read().splitlines(),
    version="0.0.2",
)
