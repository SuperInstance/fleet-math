from setuptools import setup, find_packages
setup(
    name="fleet-math",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["numpy"],
    python_requires=">=3.8",
)
