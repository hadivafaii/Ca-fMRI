import os
import sys
from os.path import join as pjoin

git_dir = pjoin(os.environ['HOME'], 'Dropbox/git')
pd_path = pjoin(git_dir, 'network-portrait-divergence')
sys.path.insert(0, pd_path)
