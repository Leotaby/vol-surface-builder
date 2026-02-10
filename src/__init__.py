"""
vol-surface-builder
====================
End-to-end implied volatility surface construction from option chains.

Modules:
    black_scholes      - Pricing, greeks, implied vol inversion
    data_feed          - Option chain retrieval and cleaning
    surface_builder    - Grid interpolation and surface construction
    visualization      - 2D/3D charting (matplotlib + plotly)
    svi_calibration    - SVI parameterization for synthetic generation
    config             - Global constants and defaults
"""

__version__ = "0.2.1"
__author__ = "Leo"
