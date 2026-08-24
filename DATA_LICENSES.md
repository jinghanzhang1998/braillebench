# Data sources, attribution, and terms

This file documents the upstream sources used to construct BrailleBench. The Braille
translations are transformations of upstream questions and answers; distributing a transformed
copy does not erase the original rights or attribution requirements.

BrailleBench 0.1.0 includes the complete set of translated records for reproducible research.
The MIT License at the repository root covers BrailleBench-authored code, not third-party source
questions or answers. Braille conversion does not replace or broaden upstream permissions.

## Included datasets

| Dataset | Upstream source | Upstream terms and release scope |
|---|---|---|---|
| GSM8K | [OpenAI grade-school-math](https://github.com/openai/grade-school-math) | The official repository uses the MIT License. The upstream notice is reproduced in `LICENSES/GSM8K-MIT.txt`. |
| AIME 2024 | Mathematical Association of America | The 30 translated 2024 competition problems are included for research evaluation and attribution. BrailleBench does not claim ownership of or grant additional rights to the source problems. |
| CommonsenseQA | [Official repository](https://github.com/jonathanherzig/commonsenseqa) | The official repository provides downloads and citation information but does not identify a dataset license. The translated development records are included for research reproducibility and are not relicensed by BrailleBench. |
| HotpotQA | [Official site](https://hotpotqa.github.io/) | Distributed under CC BY-SA 4.0. The translated records are an adaptation and retain that attribution and share-alike scope. |
| 2WikiMultiHopQA | [Official repository](https://github.com/Alab-NII/2wikimultihop) | Distributed under Apache License 2.0. A copy is included in `LICENSES/2WikiMultiHopQA-Apache-2.0.txt`. |

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for a compact mapping from packaged paths to
sources, notices, and license references. Users remain responsible for checking upstream terms for
their intended use, especially redistribution or commercial use of source content for which no
explicit open dataset license is identified.

## Dataset citations

### GSM8K

```bibtex
@article{cobbe2021gsm8k,
  title={Training Verifiers to Solve Math Word Problems},
  author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and Hesse, Christopher and Schulman, John},
  journal={arXiv preprint arXiv:2110.14168},
  year={2021}
}
```

### CommonsenseQA

```bibtex
@inproceedings{talmor2019commonsenseqa,
  title={CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge},
  author={Talmor, Alon and Herzig, Jonathan and Lourie, Nicholas and Berant, Jonathan},
  booktitle={Proceedings of NAACL-HLT},
  year={2019}
}
```

### HotpotQA

```bibtex
@inproceedings{yang2018hotpotqa,
  title={HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering},
  author={Yang, Zhilin and Qi, Peng and Zhang, Saizheng and Bengio, Yoshua and Cohen, William W. and Salakhutdinov, Ruslan and Manning, Christopher D.},
  booktitle={Proceedings of EMNLP},
  year={2018}
}
```

### 2WikiMultiHopQA

```bibtex
@inproceedings{ho2020constructing,
  title={Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps},
  author={Ho, Xanh and Nguyen, Anh-Khoa Duong and Sugawara, Saku and Aizawa, Akiko},
  booktitle={Proceedings of COLING},
  year={2020}
}
```

## liblouis

Braille translation and back-translation use [liblouis](https://github.com/liblouis/liblouis)
and its UEB tables. Liblouis is distributed under LGPL-2.1-or-later for the library, with
additional licenses for command-line tools. The release package links to but does not bundle
the liblouis native library or tables.

## BrailleBench-authored code

BrailleBench-authored source code, tests, and original documentation are licensed under the MIT
License in the repository-root `LICENSE` file. That license does not apply to third-party source
questions and answers embedded in translated benchmark records.
