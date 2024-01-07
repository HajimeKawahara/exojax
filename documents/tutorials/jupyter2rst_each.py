"""

 python jupyter2rst_each.py file
no executable.

"""

import subprocess
import pandas as pd
import sys
if True:
    ipynb_name = sys.argv[1]
    print(ipynb_name)
    subprocess.run("jupyter nbconvert --to rst "+ '"'+ipynb_name+'"',shell=True)
