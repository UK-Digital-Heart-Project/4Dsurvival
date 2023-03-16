from setuptools import setup, find_namespace_packages


setup(
    name="survival4d",
    use_scm_version=True,
    author="Declan Oregan",
    author_email="declan.oregan@lms.mrc.ac.uk",
    description="",
    packages=find_namespace_packages(),
    setup_requires=["setuptools >= 40.0.0"],
    include_package_data=True,
)
