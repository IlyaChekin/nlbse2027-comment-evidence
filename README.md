# NLBSE 2027 multitask code classification

[Paper (PDF)](paper/main.pdf) · [Round 38 model and checksums](https://github.com/IlyaChekin/nlbse2027-comment-evidence/releases/tag/v1.0.0)

The tool accepts CSV with a `Function` column and emits two binary columns, `vulnerability` and `MAT`. The released configuration is Round 38. Its frozen public predictions have F1 scores of 0.954278 and 0.935722, with mean F1 0.945000 across 30,655 functions. Its local competition score is 0.964721 on RTX 4090. The source, model archive and verification steps are described in [REPRODUCE.md](REPRODUCE.md).

The artifact includes the training and evaluation scripts, the inputs for refitting the released paired correction, and measured results for the examination of omitted comments and the matched 64-token windows. The data manifest pins the Hugging Face dataset and its hashes. The organizer's hidden test data is held by the organizers.

Third-party provenance and terms are listed in [THIRD_PARTY.md](THIRD_PARTY.md). See the [official competition page](https://nlbse2027.github.io/tools/#code-comment-classification) and [evaluation instructions](https://github.com/nlbse2027/code-comment-classification#evaluation).

The [MIT license](LICENSE) applies to the original software in this repository. See [THIRD_PARTY.md](THIRD_PARTY.md) for dataset, model, and third-party terms.
