5-6 paragraphs in total
1 diagram, 1 most interesting plot (Figure 1, Figure 2)
quotes, key findings bigger

talk about reproducibility (include link to original paper) and anomaly detection
start with methodology recap and main paper reproduced results, then exploration into new results 




# CWoLa detects stellar streams with coordinate-dependent anomaly detection

6 paragraphs:

The machine learning technique Classification Without Lables evidences the power of anomaly detection in potentially unlabled class mixtures. Following its [original development](https://link.springer.com/article/10.1007/JHEP10(2017)174) in 2017 for application to the Large Hadron Collider, Dr. Mariel Pettee and her team produced the [first application of CWoLa to astronomical data](https://academic.oup.com/mnras/article/527/3/8459/7452899) in 2023 for the detection of stellar streams. Under her supervision, I explored the scientific reproducibility of their results and the impact of coordinate choice on the performence of CWoLa for anomaly detection.


### Why stellar streams?

Stellar streams are long ribbon-like star formations that are formed from dwarf galaxies and globular clusters, whose gravitational bindings are broken apart by tidal disruptions of a host galaxy during orbit. These streams are increasingly sought for discovery as studies help reveal further illustration of the formation of galaxies, the gravitational potential of the host galaxy, and the distribution of dark matter in the universe. Because the constituent stars of a stream are of the same progenitor, a dwarf galaxy or globular gluster, detection of stellar streams can focus on grouping their similar characteristics. One important characteristic of stellar streams is that they are kinematically cold, which means that the stars have very similar proper motions.  The localization of proper motion for stellar streams justifies the understanding of a stellar stream as an anomaly, with overdense concentration of stream stars compared to background stars in proper motion space. 


### How do we detect stellar streams with CWoLa?


CWoLa is a technique for a binary classifier, which takes in two mixed samples of data and learns to distinguish individual events as likely to have come from mixed sample 1 or mixed sample 2. All events within the signal-enriched sample are assigned a proxy label of signal, while all events within the signal-depleted sample are assigned a proxy label of background. After labeling, the CWoLa paradigm involves training fully supervised classifiers to distinguish events from the two regions. Despite not seeing the true signal/background event labels, this classification between mixed samples is proven to be equivalent to the optimal classifier for distinguishing signal and background events. The model returns a value between 0 and 1 that represents how likely the event is to be from the signal region. To determine anomalies, a cut is made on the dataset ordered by highest to lowest anomaly score to select a sample with the highest ratio of signal to noise. When applied to stellar stream detection, the signal and sideband regions are constructed relative to the distribution of the localized proper motion coordinate within each data patch. The exact proprtions of stream to background stars are not required for CWoLa; the only expectation is that there is a higher proportion of stream to background stars within the signal region, with statistically identical events across both regions. This requirement is achieved by defining the signal region as ±1σ away and the sideband region as from ±1σ to ±3σ away from the median proper motion of stream stars within the data patch.

*** insert cwola diagram here
Figure 1 : The CWoLa framework is illustrated above. Classification between the mixed samples is equivalent to the optimal classifier for distinguishing between signal and background events, thanks to the [Neyman-Pearson lemma](https://royalsocietypublishing.org/rsta/article/231/694-706/289/40532/IX-On-the-problem-of-the-most-efficient-tests-of) The graph on the right shows the definition of the signal and sideband regions over a localized feature such that the signal region contains more signal events, indicating the localized anomaly. Image: [Original Astro-CWola paper](https://academic.oup.com/mnras/article/527/3/8459/7452899)



### What does the CWoLa paradigm have to do with coordinate choice?


The CWoLa paradigm for stellar streams uses the proper motion-defined signal and sideband regions to distinguish between stellar streams stars and background stars, based on their other uncorrelated characteristics. Importantly, the stellar data contains two position coordinates, two photometric features, and two proper motion coordinates. 


The original paper utilized μ_λ, proper motion in the declination direction, to define the signal and sideband search regions for training. This definition affects the events and patterns that the model sees. Therefore, the model’s degree of classification of a star as anomalous or not is dependent on the coordinates given. 

So the question arises, what would happen if we were to change the proper motion coordinate? In principle, stellar streams are localized in both proper motion coordinates, and therefore the results should hold if we were to switch that coordinate. I tested that by defining a new signal and sideband region within each patch of data based on the distribution of mu phi cos lambda, or proper motion in the direction of right ascension, and retraining the model.









full results (reproduced purity, differences with other parameter) with ratios -- plot



future study, optimization






