## A Deep Learning Framework for Predictive Maintenance of Chemical Plant Faults via Temporal Feature Engineering

## Overview
This project introduces a robust, deep learning model designed to perform continuous anomaly detection and root-cause fault diagnosis in highly complex, non-stationary chemical manufacturing environments. By addressing severe multicollinearity, prolonged mechanical faults, and the masking effects of internal PID control loops, the model successfully prevents alarm fatigue while accurately categorizing structural system failures.

## The Dataset
The analysis is grounded in the **Tennessee Eastman Process (TEP) benchmark dataset**, a widely recognized simulation of a continuous multi-unit industrial chemical plant. The dataset comprises over 10 million chronological records sampled at 3-minute intervals across 52 continuous process variables (sensors and actuators).

Prior to modeling, a comprehensive Exploratory Data Analysis (EDA) was conducted. This included autocorrelation analysis to prove temporal memory, correlation shift scoring to map structural disruptions, and an L1-regularized (LASSO) logistic regression that aggressively reduced the dataset to the top 10 most critical causal features (e.g., specific flow valves and purge compositions).

## Modeling 

**Sequential Degradation Tracking (Trans-LSTM + State-Aware CUSUM):** To extend anomaly detection into ongoing degradation monitoring, the Trans-LSTM first models the long-term sequential trends of the plant to calculate continuous reconstruction errors. A custom State-Aware Adaptive CUSUM algorithm is then applied to these errors as a dynamic gating mechanism. By locking its statistical memory during an active anomaly, this pipeline prevents the system from falsely adapting to prolonged faulty states, acting as a fatigue-resistant frontline alarm.

## Key Results
**High-Precision Frontline Alarm:** The Trans-LSTM + State-Aware CUSUM model achieved a near-perfect **Precision of 99.40%** and a PR-AUC of 0.9401, successfully ignoring minor faults masked by internal controllers to virtually eliminate false positive shutdowns.

## Repository Structure
* `data_description_&_merged.ipynb`: Understand data and clean the data.
* `EDA.ipynb`: Comprehensive Exploratory Data Analysis including baseline stability profiling, correlation shift heatmaps, and PyTorch-based LASSO dimensionality reduction.
* `model.ipynb`: Implementation of the Semi-Supervised Trans-LSTM Autoencoder and the State-Aware Adaptive CUSUM gating algorithm.
