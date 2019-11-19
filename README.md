
## 1. Characteristics of the dataset
Item | Values 
 :- | :-
umber of Mashups | 6206
Number of APIs | 12919
Number of invocations | 13107
Average number of invocations per Mashup | 2.11
Number of called APIs | 940
Called proportion of APIs | 7.28%
NUmber of interactions |  9297
Number of labeled Mashup tags | 18601
Number of Mashup tags | 403
Number of labeled API tags | 44891
Number of API tags | 473
Average length of Mashup description | 27.63
Average length of API description | 68.98
Sparsity of Mashup-API matrix | 99.84%

Run programme [dataset_characteristics.py](dataset_characteristics.py) to generate the above statistical data.

## 2. FM based Models for Web API Recommendation
* **basicFM**: [S. Rendle, “Factorization Machines,” 2010 IEEE International Conference on Data Mining, 2010, pp. 995-1000.](references/2010_ICDM_Factorization_Machines.pdf)
  * B. Cao, B. Li, J. Liu, M. Tang, and Y. Liu, “Web APIs Recommendation for Mashup Development based on Hierarchical Dirichlet Process and Factorization Machines,” International Conference on Collaborative Computing: Networking, Applications and Worksharing, 2016, pp. 3-15.
  * B. Cao, B. Li, J. Liu, M. Tang, Y. Liu, and Y. Li, “Mobile Service Recommendation via Combining Enhanced Hierarchical Dirichlet Process and Factorization Machines,” Mobile Information Systems, vol. 2019, 2019.

* **DeepFM**: [H. Guo, R. Tang, Y. Ye, Z. Li, and X. He, “DeepFM: A Factorization-Machine based Neural Network for CTR Prediction,” International Joint Conference on Artificial Intelligence, 2017, pp. 1725-1731.](references/2017_IJCAI_DeepFM_a_factorization-machine_based_neural_network_for_CTR_prediction.pdf)
  * X. Zhang, J. Liu, B. Cao, Q. Xiao, and Y. Wen, “Web Service Recommendation via Combining Doc2Vec-based Functionality Clustering and DeepFM-Based Score Prediction,” 2018 IEEE Intl Conf on Parallel & Distributed Processing with Applications, Ubiquitous Computing & Communications, Big Data & Cloud Computing, Social Computing & Networking, Sustainable Computing & Communications, 2018, pp. 509-516.

* **AFM**: [J. Xiao, H. Ye, X. He, H. Zhang, F. Wu, and T.-S. Chua, “Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks,” International Joint Conference on Artificial Intelligence, 2017, pp. 3119-3125.](references/2017_IJCAI_Attentional_factorization_machines_Learning_the_weight_of_feature_interactions_via_attention_networks.pdf)
  * Y. Cao, J. Liu, M. Shi, B. Cao, T. Chen, and Y. Wen, “Service Recommendation Based on Attentional Factorization Machine,” International Conference on Services Computing, Year, pp. 189-196.

## 3. NAFM: Neural and Attentional Factorization Machine for Web API Recommendation
* **To be added**

## 4. Neural Network Architechtures for FM based Models
Models | Architechtures 
 :- | :-:
**basicFM** | ![basicFM](neural_network_architechtures/basicFM.jpg)
**DeepFM** | ![basicFM](neural_network_architechtures/DeepFM.jpg)
**AFM** | ![basicFM](neural_network_architechtures/AFM.jpg)
**NAFM** | ![basicFM](neural_network_architechtures/NAFM.jpg)