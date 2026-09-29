# Data provenance and redistribution

## NOAA CoralTemp v3.1

The Hainan and Great Barrier Reef datasets are derived from NOAA Coral Reef
Watch CoralTemp v3.1 daily 5 km SST. NOAA states that Coral Reef Watch web
content is public domain and may be distributed freely with appropriate
credit. Cite NOAA Coral Reef Watch and Skirving et al. (2020),
`https://doi.org/10.3390/rs12233856`.

Source portal:
`https://coralreefwatch.noaa.gov/product/5km_v3.1_op/index_5km_sst.php`

## AIMS reef anchors

Great Barrier Reef anchor locations were drawn from public Australian
Institute of Marine Science monitoring resources. AIMS identifies CC BY as its
standard public-data license. Credit the Australian Institute of Marine
Science and the Long-Term Monitoring Program.

## Copernicus Marine OSTIA

The Hainan OSTIA dataset is a processed subset of the Copernicus Marine product
`SST_GLO_SST_L4_REP_OBSERVATIONS_010_011`, dataset
`METOFFICE-GLO-SST-L4-REP-OBS-SST`, version `202003`, DOI
`https://doi.org/10.48670/moi-00168`. When redistributing or publishing derived
data, use the acknowledgement required by the current Copernicus Marine
Licence Agreement.

Recommended acknowledgement form:

> Generated using E.U. Copernicus Marine Service Information;
> https://doi.org/10.48670/moi-00168.

## Processed arrays

Each NPZ file stores daily SST anomalies, a training-only climatology or the
information needed to reconstruct it, training-only anomaly scaling, DAP
coefficients, frozen split indices, dates, and the ordered pixel inventory.
No validation, development, or 2026 observation contributes to the stored
climatology, scale, or DAP coefficients.

The processed arrays are included to make peer review independent of provider
availability. Their SHA-256 hashes are recorded in `MANIFEST.sha256`.
