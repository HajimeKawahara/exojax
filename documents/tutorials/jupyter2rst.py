import subprocess
import pandas as pd

dat = pd.read_csv("list.dat", names=("ipynb", ))

for ipynb_name in dat["ipynb"]:
    print(ipynb_name)
    subprocess.run(
        "jupyter nbconvert --to notebook --inplace --ExecutePreprocessor.timeout=600  --execute "
        + '"'+ipynb_name+'"',
        shell=True)
    subprocess.run("jupyter nbconvert --to rst "+ '"'+ipynb_name+'"',shell=True)
