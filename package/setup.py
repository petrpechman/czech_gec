from setuptools import setup, find_namespace_packages

with open("README.md") as fr:
    long_description = fr.read()

setup(
    name='petr-retag',
    version='0.0.1',
    packages=find_namespace_packages(include=["src.*"]),
    author='Pechman Petr',
    description="TODO.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://todo.com',
    python_requires=">=3.9",
    package_data={
        'src.retag.morphodita': ["*.dict"],
        'src.retag.vocabularies' : ["*.tsv"],
    },
    include_package_data=True,
    entry_points="""
        [console_scripts]
        retag=src.utils.retag:main_cli
        create_errors=src.utils..create_errors:main_cli
    """,
    install_requires=[
        'setuptools',
    ],
)
