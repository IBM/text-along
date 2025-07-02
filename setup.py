import io
import re

from pkg_resources import parse_requirements
from setuptools import find_packages, setup

# get pip install dependencies from requirements.txt
vcs = re.compile(r"(git|svn|hg|bzr)\+")

REQUIREMENTS = []
try:
    with open("requirements.txt") as fp:
        print("open requirements.txt")
        for requirement in parse_requirements(fp):
          #print(str(requirement))
          # At the moment ibm_watson seems to break pip install, so remove.
          if str(requirement) != 'ibm-watson':
            REQUIREMENTS.append(str(requirement))
        # does not seem to procuce anything
        VCS_REQUIREMENTS = [
            str(requirement)
            for requirement in parse_requirements(fp)
            if vcs.search(str(requirement))
        ]
except FileNotFoundError:
    # requires verbose flags
    print("requirements.txt not found.")
    REQUIREMENTS = []
    VCS_REQUIREMENTS = []

# VCS_REQUIREMENTS.append('ibm-watson') Need ibm-watson, but fails to install when testing 
# when part of the requirements within the package. 
print(REQUIREMENTS)
print(VCS_REQUIREMENTS)

# get module version number from t4jbias-backend/__init__.py
vmatch = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
    io.open("__init__.py", encoding="utf_8_sig").read(),
)
if vmatch is None:
    raise SystemExit("Version number not found.")
__version__ = vmatch.group(1)

# get module name from t4jbias-backend/__init__.py
nmatch = re.search(
    r'__name__\s*=\s*[\'"]([^\'"]*)[\'"]',
    io.open("__init__.py", encoding="utf_8_sig").read(),
)
if nmatch is None:
    raise SystemExit("Module name not found.")
__name__ = nmatch.group(1)

print(__name__)

# run setup
setup(
    name=__name__,
    version=__version__,
    author="IBM Tech for Justice - Media Bias Team",
    author_email="lamogha.chiazor@ibm.com",
    url="https://github.ibm.com/T4J-Media-Bias-Analysis/text-along",
    license="Apache-2.0 license",
    packages=find_packages(),
    package_data={"t4jbias_news":
        ["*.txt",
        "Version"
        ]},
    description='T4JBIAS-news python package.',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=REQUIREMENTS,
    extras_require={"vcs": VCS_REQUIREMENTS},
    python_requires='>=3.7,<4.0'
)