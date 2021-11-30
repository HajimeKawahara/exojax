# How to contribute to exojax

## Branch model and pull-request (PR)

We adopt a simple git-flow model, consisting of master, develop, contributor-defined branches. PR by contributors via a folked repository is always welcome.

- master: stable branch. Once a set of functions is implemented in the **develop** branch, the maintainer (@HajimeKawahara) will merge them to the master branch through the release process (release branch).

- develop: default branch for developers. PR by contributors will be merged to this branch, after the review process.


## Issues and Discussion

You can ask anything about exojax in Issue tracker and Discussion. 

## Tests

As proposed by @gully #86, we now have unit tests using pytest. 
Please include unit tests if your update gives new functionality to exojax.
Some external files are needed to complete all the tests. Check [here](http://secondearths.sakura.ne.jp/exojax/data/).

## TIPS

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

