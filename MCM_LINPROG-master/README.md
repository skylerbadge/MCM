# Minimal complexity machines
The VC dimension measures the complexity of a learning machine, and a low VC dimension leads to good generalization. While SVMs produce state-of-the-art learning performance, it is well known that the VC dimension of a SVM can be unbounded; despite good results in practice, there is no guarantee of good generalization. In this paper, we show how to learn a hyperplane classifier by minimizing an exact, or Θ bound on its VC dimension. The proposed approach, termed as the Minimal Complexity Machine (MCM), involves solving a simple linear programming problem. Experimental results show, that on a number of benchmark datasets, the proposed approach learns classifiers with error rates much less than conventional SVMs, while often using fewer support vectors. On many benchmark datasets, the number of support vectors is less than one-tenth the number used by SVMs, indicating that the MCM does indeed learn simpler representations.

## Code
The code is written in Matlab, and uses linprog to solve the optimization problem

## Examples
Here we present the algorithms and 22 UCI datasets, for users to test their code on.


## Citation
If you use the code please cite the following papers using the bibtex entry:

```
@article{Jayadeva:2015:LHC:2841459.2841848,
 author = {Jayadeva},
 title = {Learning a Hyperplane Classifier by Minimizing an Exact Bound on the VC Dimension1},
 journal = {Neurocomput.},
 issue_date = {February 2015},
 volume = {149},
 number = {PB},
 month = feb,
 year = {2015},
 issn = {0925-2312},
 pages = {683--689},
 numpages = {7},
 url = {http://dx.doi.org/10.1016/j.neucom.2014.07.062},
 doi = {10.1016/j.neucom.2014.07.062},
 acmid = {2841848},
 publisher = {Elsevier Science Publishers B. V.},
 address = {Amsterdam, The Netherlands, The Netherlands},
 keywords = {Complexity, Generalization, Machine learning, Sparse, Support vector machines, VC dimension},
} 

@article{chandra2014learning,
  title={Learning a hyperplane regressor by minimizing an exact bound on the VC dimension},
  author={Chandra, Suresh and Sabharwal, Siddarth and Batra, Sanjit S and others},
  journal={arXiv preprint arXiv:1410.4573},
  year={2014}
}

```

## Research Paper
The papers for the same is available at:

http://www.sciencedirect.com/science/article/pii/S0925231214010194
http://www.sciencedirect.com/science/article/pii/S092523121500939X
