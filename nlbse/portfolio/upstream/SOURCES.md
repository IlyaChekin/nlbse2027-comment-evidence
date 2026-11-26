# Upstream pretrained weights

The bundled CodeBERT and CodeT5+ checkpoints are fine-tuned derivatives. Upstream licenses are included with attribution.

- [Microsoft CodeBERT model card](https://huggingface.co/microsoft/codebert-base) and [MIT license](https://github.com/microsoft/CodeBERT/blob/master/LICENSE).
- [Salesforce CodeT5+ 220M model card](https://huggingface.co/Salesforce/codet5p-220m) and [BSD 3-Clause license](https://github.com/salesforce/CodeT5/blob/main/LICENSE.txt).
- [Microsoft GraphCodeBERT model card](https://huggingface.co/microsoft/graphcodebert-base). Its weights are used for the pilot and retrieved separately from the revision pinned in `fetch_backbones.sh`.

The official NLBSE dataset is fetched from the revision in `nlbse/data/manifest.json`.
