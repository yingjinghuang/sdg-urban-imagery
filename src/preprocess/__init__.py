"""Country-level preprocessing.

The raw census / indicator / boundary tables differ in source and schema
across the seven countries (US ACS, AU ABS, BR IBGE, CH HK C&SD, FR INSEE,
PT INE, NG custom), so per-country ingestion is intrinsically heterogeneous.
The processed outputs (``labels.pkl``, ``labels_norm.pkl``, ``paths.pkl``,
``geo.pkl``) are uniform across countries and are distributed via the
public Zenodo deposit — users reproducing only the model/figures do not
need to re-run this stage.

Scripts in this module cover the parts of the preprocess pipeline that
*are* portable across countries: bounding-box tile generation for
satellite downloads, and the canonical label-scaling/normalization step.
"""
