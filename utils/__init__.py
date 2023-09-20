import os
import sys
from os.path import join as pjoin

git_dir = pjoin(os.environ['HOME'], 'Dropbox/git')
br_path = pjoin(git_dir, 'brainrender')
vedo_path = pjoin(git_dir, 'vedo')
sys.path.insert(0, br_path)
sys.path.insert(0, vedo_path)
