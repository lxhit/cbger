# Public release checklist

- [ ] Replace the placeholder project author in `CITATION.cff` and README BibTeX.
- [ ] Select and add an explicit code license.
- [ ] Confirm MicroLens and FineVideo derived-metadata/feature redistribution terms.
- [ ] Decide whether 85 MB CLIP features remain in Git or move to Git LFS/Hugging Face.
- [ ] Add exact QD-DETR commit hash and publish the MQVTG compatibility patch/codebook.
- [ ] Run `pytest -q`, `sha256sum -c checksums.sha256`, and a fresh three-seed training smoke test.
- [ ] Verify no access tokens, cookies, user-local paths or private source media are included.
