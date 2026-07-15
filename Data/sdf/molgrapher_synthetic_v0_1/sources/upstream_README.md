---
dataset_info:
  features:
  - name: id
    dtype: int64
  - name: image
    dtype: image
  - name: mol
    dtype: string
  - name: smiles
    dtype: string
  - name: keypoints
    sequence: int64
  splits:
  - name: train
    num_bytes: 10662864691.988
    num_examples: 219068
  - name: validation
    num_bytes: 1332412015.82
    num_examples: 27383
  - name: test
    num_bytes: 1352273440.768
    num_examples: 27384
  download_size: 13043619469
  dataset_size: 13347550148.576
---
# MolGrapher-Synthetic-300K

MolGrapher-Synthetic-300K is the synthetic dataset introduced in [MolGrapher: Graph-based Visual Recognition of Chemical Structures](https://github.com/DS4SD/MolGrapher).

Our dataset is created using molecule SMILES retrieved from the database PubChem. Training images are then generated from SMILES using the molecule drawing library RDKit.
The synthetic training set is augmented at multiple levels:

- Molecule level: Molecules are randomly transformed by: (1) displaying explicit hydrogens, (2) reducing of the size of bonds connected to explicit hydrogens, (3) displaying explicit methyls, (4) displaying explicit carbons, (5) selecting a molecular conformation, (6) removing implicit hydrogens of atom labels, (7) rotating triple bonds,  (8) displaying explicit carbons connected to triple bonds, adding artificial superatom groups with (9) single or (10) multiple attachment points, (11) displaying wedge bonds using solid or dashed bonds, and (12) displaying single bonds as wavy bonds. 
- Rendering level: The rendering parameters used in RDKit are randomly set: (1) the bond width, (2) the font, (3) the font size, (4) the atom label padding, (5) the molecule rotation, which does not rotate atom labels, (6) the display of atom indices and (7) their font size, (8) the hand-drawing style, (9) the charges positions, (10) the display of encircled charges and (11) their size, and (12) the display of aromatic cycles using circles.

Together with the training images, we generate the graph ground-truth, i.e the graph connectivity, the atoms and bonds labels, and their positions. RDKit allows to embed to the generated image some metadata, which stores the mapping between atom indices and their positions in the image. At the same time, we store a MolFile containing the graph connectivity information and the class of each atom and bond. By combining both of them, the graph ground-truth can be re-created. 

See [dataset_explorer.ipynb](https://huggingface.co/datasets/ds4sd/MolGrapher-Synthetic-300K/blob/main/dataset_explorer.ipynb) for examples on reading samples.
