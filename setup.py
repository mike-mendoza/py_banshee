import setuptools

# Reads the content of your README.md into a variable to be used in the setup below
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='py_banshee',                           # should match the package folder
    packages=['py_banshee'],                   # should match the package folder
    version='1.1.1',                              # important for updates
    license='GNU',                                  # should match your chosen license
    description='Py_Banshee allows for quantifying non-parametric Bayesian Networks',
    long_description=long_description,              # loads your README.md
    long_description_content_type="text/markdown",  # README.md is of type 'markdown'
    author='Paul Koot, Miguel Angel Mendoza Lugo, Oswaldo Morales Napoles',
    author_email='mendozalugo@gmail.com',
    url='https://github.com/mike-mendoza/py_banshee', 
    project_urls = {                                # Optional
        "Bug Tracker": "https://github.com/mike-mendoza/py_banshee/issues"
    },
    install_requires=['ipython','pingouin','graphviz','pydot','pycopula','argparse', 'scipy', 'numpy', 'pandas', 'matplotlib', 'seaborn','networkx'],    # list all packages that your package uses
    keywords=["pypi", "py_banshee", "tutorial"], #descriptive meta-data
    classifiers=[                                   # https://pypi.org/classifiers
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Documentation',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    
    download_url="https://github.com/mike-mendoza/py_banshee/archive/refs/tags/v1.1.1.tar.gz",
)