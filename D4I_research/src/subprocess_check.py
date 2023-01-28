import subprocess
import sys

out = subprocess.run('ls', shell=True)

print(out)