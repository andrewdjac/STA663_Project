from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name = 'LDA_AandB',
    url = 'https://github.com/andrewdjac/STA663_Project',
    author = 'Andrew Cooper, Brian Cozzi',
    # Needed to actually package something
    packages = ['LDA_AandB'],
    # Needed for dependencies
    install_requires = ['numpy', 'sklearn', 'scipy', 'numba'],
    # *strongly* suggested for sharing
    version = '0.1',
    # The license can be anything you like
    license = 'MIT',
    description = 'An package implementing LDA for topic modeling using a collapsed Gibbs Sampler',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)