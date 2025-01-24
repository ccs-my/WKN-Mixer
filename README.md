# WKN-Mixer

# Title
Detection of Abnormal Temperature Increases for Dry-Type Transformers Based on Working Condition Recognition and WKN-Mixer

# Abstract
In the realm of maintaining secure and stable distribution networks, the detection of abnormal temperature rises in dry-type transformers plays a pivotal role. Conventional approaches relying on thermal-model based anomaly detection, however, struggle to accommodate the intricate and fluctuating working conditions prevalent in these transformers. This gives rise to a wide range of issues, encompassing not only limited accuracy but also a higher incidence of false alarms than is typically observed. To address this issue, a comprehensive framework for detecting abnormal temperature rises in dry-type transformers is proposed, which is based on the accurate recognition of working conditions. Firstly, the Soft-Dynamic Time Warping (Soft-DTW) approach is employed to conduct curve clustering on the working condition parameters of dry-type transformers. Through gauging the similarity of time-series data, it effectively captures the working condition features embedded in the intricate monitoring data, thereby enhancing the precision of working condition recognition. Subsequently, a WaveletKernelNet-Mixer (WKN-Mixer) is devised to precisely forecast the three-phase winding temperatures of dry-type transformers. WaveletKernelNet is seamlessly integrated into the input layer of the Mixer, enabling the network focus on features in the time-frequency domain. Additionally, the architecture employs an advanced multi-layer perceptron, adeptly capturing the intricate non-linear interactions prevalent in multivariate time-series data across both temporal and channel dimensions. This integration significantly boosts the detection accuracy of the network model, ensuring more precise results. Finally, case studies verified that the proposed method reduced the mean absolute error (MAE) by 31.9\% after recognizing working conditions. Under normal operation, the WKN-Mixer achieved the lowest FPR and an average F1-score 7\% higher than its peers at a 3.0\% anomalous level.

# Dataset Introduction
The dataset was collected from a dry-type transformer. The data collection period spans from February 4, 2022, to October 25, 2023. The original data covers 13 dimensions, specifically including three phase winding temperatures, environmental temperature, three phase currents, three phase voltages, active power, reactive power, and environmental humidity. The data sampling frequency is one data point every 5 minutes. It should be noted that there are some missing values in the original data.
