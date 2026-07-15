---
dataset_info:
  features:
  - name: filename
    dtype: string
  - name: image
    dtype: image
  - name: mol
    dtype: string
  splits:
  - name: clean
    num_bytes: 88030343.0
    num_examples: 10000
  - name: abbreviated
    num_bytes: 84064086.0
    num_examples: 10000
  - name: large
    num_bytes: 238905697.0
    num_examples: 10000
  download_size: 291334748
  dataset_size: 411000126.0
---
# USPTO-30K

USPTO-30K is the benchmark dataset introduced in [MolGrapher: Graph-based Visual Recognition of Chemical Structures](https://github.com/DS4SD/MolGrapher).

Existing benchmarks for Optical Chemical Structure Recognition have some limitations. 
Being created using only a few documents, they contain batches of very similar molecules. For example in a patent, a molecule could typically be displayed together with all the substituent of one particular substructure, resulting in large batches of almost identical molecules. 
Additionally, the existing sets contain molecules of different kinds, including superatom groups and various markush features, which should be evaluated independently.
In practice, it is important to delimit on which types of molecules models can be applied.

We introduce USPTO-30K, a large-scale benchmark dataset of annotated molecule images, which overcomes these limitations. 
It is created using the pairs of images and MolFiles by the United States Patent and Trademark Office. 
Each molecule was independently selected among all the available documents from 2001 to 2020. 
The set consists of three subsets to decouple the study of clean molecules, molecules with abbreviations and large molecules. 

- USPTO-10K contains 10,000 clean molecules, i.e. without any abbreviated groups.
- USPTO-10K-abb contains 10,000 molecules with superatom groups. 
- USPTO-10K-L contains 10,000 clean molecules with more than 70 atoms. 

[More Information needed](https://github.com/huggingface/datasets/blob/main/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)