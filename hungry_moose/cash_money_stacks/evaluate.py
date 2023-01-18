import pyfolio

with pyfolio.plotting.plotting_context(font_scale=1.1):
     pyfolio.create_full_tear_sheet(returns = ensemble_strat,
     benchmark_rets=dow_strat, set_context=False)