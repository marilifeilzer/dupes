from setuptools import find_packages, setup

with open("requirements.txt", "r") as file:
    lines = file.readlines()

reqs = [req.strip() for req in lines]

setup(name="dupes", packages=find_packages(), install_requires=reqs)