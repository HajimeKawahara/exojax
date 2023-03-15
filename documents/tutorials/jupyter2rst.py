"""

 python jupyter2rst.py [exe]
 if putting exe, it executes all the notebooks.

"""

import subprocess
import pandas as pd
import sys
dat = pd.read_csv("list.dat", names=("ipynb", ))
#dat = pd.read_csv("listexe.dat", names=("ipynb", ))

print(sys.argv[1])
for ipynb_name in dat["ipynb"]:
    print(ipynb_name)
    if sys.argv[1] == "exe":
        subprocess.run(
            "jupyter nbconvert --to notebook --inplace --ExecutePreprocessor.timeout=600  --execute "
            + '"'+ipynb_name+'"',
            shell=True)
    subprocess.run("jupyter nbconvert --to rst "+ '"'+ipynb_name+'"',shell=True)
