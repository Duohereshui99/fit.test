Create eye-pleasing cornerplots (and maybe other types of plots in the future).

Usage example(s):

  import numpy as np
  import prettyplease

  # Generate some example data
  rng = np.random.default_rng()
  n_samples = 100000
  n_dim = 4
  x = rng.normal(1.618, 0.2, size=(n_samples, n_dim))
  # x is an ndarray of shape (n_samples, n_dim)

  # With default settings
  fig = prettyplease.corner(x)
  fig.savefig('default.pdf', bbox_inches='tight')

  # With some customization
  labels = [rf'$\theta_{i}$' for i in range(n_dim)]
  fig = prettyplease.corner(x, bins=30, labels=labels, plot_estimates=True, colors='red', n_uncertainty_digits=4, title_loc='center', figsize=(7,7))
  fig.savefig('custom.pdf', bbox_inches='tight')
