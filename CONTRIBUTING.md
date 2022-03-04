# How to contribute to ExoJAX

## Branch model and pull-request (PR)

We adopt a simple git-flow model, consisting of master, develop, contributor-defined, and release branches. PR by contributors via a folked repository is always welcome.

- master: stable branch. Once a set of functions is implemented in the **develop** branch, the maintainer (@HajimeKawahara) will merge them to the master branch through the release process (release branch).

- develop: default branch for developers. PR by contributors will be merged to this branch, after the review process.

- contributor-defined branches: The contirbutors make their own branch originated from the develop branch and after developing, we send a PR to the develop branch. The branch name is like the followings: feature/hminus_opacity, bugfix/hminus, etc.

- release: we make the release branch prior to a new version release. The contributors are expected to review the release branch.  


## Issues and Discussion

You can ask anything about exojax in Issue tracker and Discussion. 

## Tests

As proposed by @gully #86, we now have unit tests using pytest. 
Please consider to include unit tests if your update gives new functionality to ExoJAX.
Some external files are needed to complete all the tests. Check [here](http://secondearths.sakura.ne.jp/exojax/data/).

## TIPS

### pyformat

Pyformat formats the code to follow a consistent style that used in ExoJAX.

At minimum, you can use it as

```sh
pyformat -i your_code.py
```

### How to include data into exojax

- Edit MANIFEST.in
- Put data files in data/somewhere/
- Use pkgutil to load them.

For instance,

```python
import pandas as pd
import pkgutil
from io import BytesIO

adata = pkgutil.get_data('exojax',"data/somewhere/hoge.txt")
dat = pd.read_csv(BytesIO(adata), sep="\s+")
```

