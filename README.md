# A generative model for proposing drug candidates satisfying anticancer property using conditional variational autoencoder
#### Sunghoon Joo, Min Soo Kim*, Jaeho Yang, and Jaehyun Park
#### AI advanced research laboratory, Samsung SDS. 56, Sungchon-gil, Seocho-gu, Seoul 06765, Republic of Korea.

## ABSTRACT
Deep-learning-based molecular generative models have successfully identified drug candidates with desired properties against biological targets of interest. However, syntactically invalid molecules generated from a deep-learninggenerated model hinders the model from being applied to drug
discovery. Herein, we propose a conditional variational autoencoder (CVAE) as a generative model to propose drug candidates with the desired property outside a dataset range. We train the CVAE using molecular fingerprints and corresponding
GI50 (inhibition of growth by 50%) results for breast cancer cell lines instead of training with various physical properties for each molecule together. We confirm that the generated fingerprints, not included in the training dataset, represents the desired property
using the CVAE model. In addition, our method can be used as a query expansion method for searching databases because fingerprints generated using our method can be regarded as expanded queries.
