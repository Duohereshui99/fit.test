# Copyright 2021-2024 Isak Svensson

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

#!/usr/bin/env python3

from setuptools import setup
import subprocess

cmd = ['git', 'describe', '--tags', '--abbrev=0']
version = subprocess.run(cmd, stdout=subprocess.PIPE)
version = version.stdout.decode('utf-8').strip()

setup(
    name='prettyplease',
    version=version,
    description='Create plots.',
    author='Isak Svensson',
    author_email='isak.svensson@chalmers.se',
    packages=['prettyplease'],
    install_requires=['numpy'],
)
