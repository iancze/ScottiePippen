#!/usr/bin/env python

import yaml
import triangle
import numpy as np

f = open("config.yaml")
config = yaml.load(f)
f.close()

flatchain = np.load(config["TlL_samples"])


labels = [r"$T_\textrm{eff}$ [K]", r"$\log_{10} L\; [L_\odot]$"]

figure = triangle.corner(flatchain, quantiles=[0.16, 0.5, 0.84],
    plot_contours=True, plot_datapoints=False, labels=labels, show_titles=True)
figure.savefig("triangle_TLl.png")
