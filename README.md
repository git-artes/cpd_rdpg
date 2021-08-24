# cpd_rdpg: Change-Point Detection on Random-Dot Product Graphs (RDPG)

In this repository we share (and keep) implementations of some algorithms regarding the problem of Change-Point Detection (CPD) on graphs. We are particularly interested in the very versatile RDPG model for random graphs (see for instance, [this very complete survey](https://dl.acm.org/doi/abs/10.5555/3122009.3242083)). As of this moment, we have two algorithms: 
- The offline CPD algorithm described in ``Change point localization in dependent dynamic nonparametric random dot product graphs'' by Oscar Hernan Madrid Padilla, Yi Yu, Carey E. Priebe (preprint available at https://arxiv.org/abs/1911.07494). This is implemented in `cpd.py`. 
- The online CPD algorithm described in ``Online Change Point Detection for Random Dot Product Graphs'' by Bernardo Marenco, Paola Bermolen, Marcelo Fiori, Federico Larroca and Gonzalo Mateos. Available soon. This is implemented in `cpd_online.py`. 

It depends on libraries typically available on most modern Python distributions (in particular NetworkX), with the exception of Graspologic. Visit https://graspologic.readthedocs.io/. 

We have included the main modules and some simple examples we've used to generate some of the graphs of the associated papers. Additionally, we have included some examples that use real datasets and both online and offline methods: 

- `cpd_example_football_conmebol.py`. It considers graphs with the yearly number of matches played between football national teams of South America. The dataset was obtained from [https://www.eloratings.net/](https://www.eloratings.net/). You may download the csv we prepared from [this link](https://www.fing.edu.uy/owncloud/index.php/s/V2tk4MxZxAvNidx/download).

