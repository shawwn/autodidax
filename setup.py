# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['autodidax']

package_data = \
{'': ['*']}

setup_kwargs = {
    'name': 'autodidax',
    'version': '0.1.0',
    'description': '',
    'long_description': '# autodidax\n',
    'author': 'Shawn Presser',
    'author_email': 'shawnpresser@gmail.com',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'python_requires': '>=3.9,<4.0',
}


setup(**setup_kwargs)
