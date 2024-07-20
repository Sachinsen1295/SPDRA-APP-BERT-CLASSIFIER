import setuptools

_version_ = '0.0.1'
REPO_NAME = 'SPDRA_APP' 
AUTHOR_NAME = 'Sachinsen1295'
SOURCE_REPO = "BERTCLASSIFIER"
AUTHOR_EMAIL = "sachin.sen1295@gmail.com"



setuptools.setup(
                 name=SOURCE_REPO,
                 version=_version_,
                 author=AUTHOR_NAME,
                 author_email=AUTHOR_EMAIL,
                 description="This is my BERT classification Project",
                 url=f"https://github.com/{AUTHOR_NAME}/{REPO_NAME}",
                
                      package_dir={"":"src"},
                      packages=setuptools.find_packages(where="src")
                 )