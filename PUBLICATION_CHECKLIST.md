# GitHub publication checklist

The generated archives are technically validated release candidates. Complete these ownership
and rights steps before making the repository public:

- [ ] Select a license for BrailleBench-authored code and add its full text as `LICENSE`.
- [ ] Confirm redistribution permission for AIME 2024-derived records, or remove
      `data/braille/aime24/` from the public repository.
- [ ] Confirm redistribution terms for CommonsenseQA-derived records, or remove
      `data/braille/commonsenseqa/` from the public repository.
- [ ] Include the required MIT, CC BY-SA 4.0, and Apache 2.0 notices for redistributed datasets.
- [ ] Add the BrailleBench paper/authors/DOI or repository citation to the README and optionally
      create `CITATION.cff`.
- [ ] Replace any placeholder repository URLs after the GitHub repository is created.
- [ ] Run `python src/validate_release.py --check-manifest DATA_MANIFEST.json` from a fresh clone.
- [ ] Run the regression suite in an environment with real liblouis Python bindings and UEB tables.
- [ ] Run a one-record model smoke test and inspect its JSONL output before tagging a release.

Do not commit API keys, provider profiles, cluster scripts, internal logs, paper review
correspondence, or historical result archives to the public repository.
