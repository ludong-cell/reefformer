# Public release checklist

- [ ] Replace the neutral contributor name in `CITATION.cff` with approved authors.
- [x] Insert the final GitHub URL.
- [ ] Insert the archived release DOI.
- [ ] Insert the corresponding-author contact address.
- [ ] Confirm all authors approve the MIT software license.
- [ ] Confirm that the processed OSTIA subset may be redistributed under the
      current Copernicus Marine agreement; otherwise move it to a data DOI or
      replace it with a download-and-build script.
- [ ] Run `pytest` in a fresh environment.
- [x] Run all three reference reproduction commands.
- [x] Run the sensitive-information audit.
- [ ] Create a signed version tag and an immutable archival release.
