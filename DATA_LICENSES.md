# Data licenses and attribution

This file documents the upstream sources used to construct BrailleBench. The Braille
translations are transformations of upstream questions and answers; distributing a transformed
copy does not erase the original rights or attribution requirements.

This is a release audit, not legal advice. Verify the terms with your institution or the
rightsholder before publishing the data.

## Publication status

| Dataset | Upstream source | Upstream terms found | Public redistribution status |
|---|---|---|---|
| GSM8K | [OpenAI grade-school-math](https://github.com/openai/grade-school-math) | MIT license in the official repository | License text/notice must accompany redistributed material |
| AIME 2024 | Mathematical Association of America | Competition problems are copyrighted; no open dataset license was identified | **Hold** until written permission or another documented legal basis is obtained |
| CommonsenseQA | [Official repository](https://github.com/jonathanherzig/commonsenseqa) | The official repository provides data links and citation information but no license file was found | **Hold** until the authors/rightsholder confirm redistribution terms |
| HotpotQA | [Official site](https://hotpotqa.github.io/) | Dataset distributed under CC BY-SA 4.0 | Attribute and distribute adaptations under compatible share-alike terms |
| 2WikiMultiHopQA | [Official repository](https://github.com/Alab-NII/2wikimultihop) | Apache License 2.0 | Include the Apache 2.0 license and required notices |

The complete local release candidate includes all five datasets so the benchmark can be
validated as a research artifact. Do not publish the AIME-derived or CommonsenseQA-derived
Braille files in a public GitHub repository until the two hold items above are resolved.

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

The repository owner still needs to select and add a project-wide code license before public
release. Without an explicit license, default copyright rules apply and outside users do not
automatically receive permission to modify or redistribute the code.
