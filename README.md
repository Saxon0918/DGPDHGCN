**Data Files**

This repository contains several CSV files that list names and interactions of diseases, drugs,genes, and proteins used in our research. Below is a description of each file:

-  disease.csv: Disease name
-  disease_disease.csv: Disease interaction
-  disease_gene.csv: Disease-gene reaction
-  drug.csv: Drug name
-  drug_drug.csv: Drug interaction
-  drug_gene.csv: Drug-gene reaction
-  gene.csv: Gene names
-  gene_gene.csv: Gene-gene reaction
-  mat_disease_disease.csv: Adjacency matrix of disease-disease similarity
-  mat_disease_drug.csv: Adjacency matrix of disease-drug association
-  mat_disease_gene.csv: Adjacency matrix of disease-gene association
-  mat_disease_protein.csv: Adjacency matrix of disease-protein association
-  mat_drug_drug.csv: Adjacency matrix of drug-drug similarity
-  mat_drug_gene.csv: Adjacency matrix of drug-gene association
-  mat_drug_protein.csv: Adjacency matrix of drug-protein association
-  mat_gene_gene.csv: Adjacency matrix of gene-gene interactions
-  mat_protein_protein.csv: Adjacency matrix of protein-protein interaction


**Code Files**

This repository also includes several Python files used for training the model. Below is a description of each script:

-  main.py: Runs the main function to train the model.
-  dataprocess.py: Constructs the data required for training the model.
-  model.py: Defines the overall framework of the model.
-  utils.py: Contains auxiliary functions necessary for training the model.