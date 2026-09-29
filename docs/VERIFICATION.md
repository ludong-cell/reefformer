# Verification record

Local release verification was completed on 2026-09-29.

| Check | Result |
|---|---|
| Editable package installation | Pass |
| Synthetic demo | Pass |
| Unit and checkpoint-loading tests | 5 passed |
| Hainan CoralTemp reference reproduction | Pass; maximum metric difference 2.7e-9 deg C |
| Great Barrier Reef CoralTemp reproduction | Pass; maximum metric difference 7.9e-10 deg C |
| Hainan OSTIA reproduction | Pass; maximum metric difference 3.0e-9 deg C |
| Sensitive-information and local-path scan | Pass |
| Files at least 50 MB | None |

The reproduction tolerance is `1e-4 deg C`. Generated prediction arrays and
reports are written under the ignored `outputs/` directory.
