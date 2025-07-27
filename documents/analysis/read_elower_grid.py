# %%
from exojax.opacity.premodit.lbderror import optimal_params
Tl_in = 500.0  #K
Tu_in = 1200.0  #K
diffmode = 2
dE, Tl, Tu = optimal_params(Tl_in, Tu_in, diffmode)
print(dE, Tl, Tu)
#750.0 1153.6267095763965 554.1714566743503
