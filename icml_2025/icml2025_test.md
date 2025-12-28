# ICML 2025 papers (OpenReview export)

- **Generated (UTC)**: 2025-12-28 12:59:32Z
- **Count**: 5

## Index

- [Emoji Attack: Enhancing Jailbreak Attacks Against Judge LLM Detection](#emoji-attack-enhancing-jailbreak-attacks-against-judge-llm-detection)
- [KGMark: A Diffusion Watermark for Knowledge Graphs](#kgmark-a-diffusion-watermark-for-knowledge-graphs)
- [LSCD: Lomb--Scargle Conditioned Diffusion for Time series Imputation](#lscd-lomb-scargle-conditioned-diffusion-for-time-series-imputation)
- [UnHiPPO: Uncertainty-aware Initialization for State Space Models](#unhippo-uncertainty-aware-initialization-for-state-space-models)
- [When Will It Fail?: Anomaly to Prompt for Forecasting Future Anomalies in Time Series](#when-will-it-fail-anomaly-to-prompt-for-forecasting-future-anomalies-in-time-series)

## Emoji Attack: Enhancing Jailbreak Attacks Against Judge LLM Detection

- **Authors**: Zhipeng Wei, Yuqi Liu, N. Benjamin Erichson
- **Venue**: ICML 2025 poster
- **Primary area**: social_aspects->safety
- **OpenReview**: `https://openreview.net/forum?id=Q0rKYiVEZq`
- **PDF**: `https://openreview.net/pdf/9c3c465a66876cc791e42e3c2a98df64e54519c6.pdf`
- **Keywords**: LLM safety; Jailbreaking Attacks; Judge LLMs; Token Segmentation

### Abstract

Jailbreaking techniques trick Large Language Models (LLMs) into producing restricted output, posing a potential threat. One line of defense is to use another LLM as a Judge to evaluate the harmfulness of generated text. However, we reveal that these Judge LLMs are vulnerable to token segmentation bias, an issue that arises when delimiters alter the tokenization process, splitting words into smaller sub-tokens. This alters the embeddings of the entire sequence, reducing detection accuracy and allowing harmful content to be misclassified as safe. In this paper, we introduce Emoji Attack, a novel strategy that amplifies existing jailbreak prompts by exploiting token segmentation bias. Our method leverages in-context learning to systematically insert emojis into text before it is evaluated by a Judge LLM, inducing embedding distortions that significantly lower the likelihood of detecting unsafe content. Unlike traditional delimiters, emojis also introduce semantic ambiguity, making them particularly effective in this attack. Through experiments on state-of-the-art Judge LLMs, we demonstrate that Emoji Attack substantially reduces the unsafe prediction rate, bypassing existing safeguards.

### BibTeX

```
@inproceedings{
wei2025emoji,
title={Emoji Attack: Enhancing Jailbreak Attacks Against Judge {LLM} Detection},
author={Zhipeng Wei and Yuqi Liu and N. Benjamin Erichson},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=Q0rKYiVEZq}
}
```

## KGMark: A Diffusion Watermark for Knowledge Graphs

- **Authors**: Hongrui Peng, Haolang Lu, Yuanlong Yu, WeiYe Fu, Kun Wang, Guoshun Nan
- **Venue**: ICML 2025 poster
- **Primary area**: social_aspects->fairness
- **OpenReview**: `https://openreview.net/forum?id=GKZySvM2t9`
- **PDF**: `https://openreview.net/pdf/a67c72895d789c5a72a9f927618fa5bccf613c6c.pdf`
- **Keywords**: Watermarking, Knowledge Graph, Diffusion Models, Generative Models

### Abstract

Knowledge graphs (KGs) are ubiquitous in numerous real-world applications, and watermarking facilitates protecting intellectual property and preventing potential harm from AI-generated content. Existing watermarking methods mainly focus on static plain text or image data, while they can hardly be applied to dynamic graphs due to spatial and temporal variations of structured data. This motivates us to propose KGMark, the first graph watermarking framework that aims to generate robust, detectable, and transparent diffusion fingerprints for dynamic KG data. Specifically, we propose a novel clustering-based alignment method to adapt the watermark to spatial variations. Meanwhile, we present a redundant embedding strategy to harden the diffusion watermark against various attacks, facilitating the robustness of the watermark to the temporal variations. Additionally, we introduce a novel learnable mask matrix to improve the transparency of diffusion fingerprints. By doing so, our KGMark properly tackles the variation challenges of structured data. Experiments on various public benchmarks show the effectiveness of our proposed KGMark.

### BibTeX

```
@inproceedings{
peng2025kgmark,
title={{KGM}ark: A Diffusion Watermark for Knowledge Graphs},
author={Hongrui Peng and Haolang Lu and Yuanlong Yu and WeiYe Fu and Kun Wang and Guoshun Nan},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=GKZySvM2t9}
}
```

## LSCD: Lomb--Scargle Conditioned Diffusion for Time series Imputation

- **Authors**: Elizabeth Fons, Alejandro Sztrajman, Yousef El-Laham, Luciana Ferrer, Svitlana Vyetrenko, Manuela Veloso
- **Venue**: ICML 2025 poster
- **Primary area**: deep_learning->sequential_models_time_series
- **OpenReview**: `https://openreview.net/forum?id=GdYg0Ohx0k`
- **PDF**: `https://openreview.net/pdf/12163661cd94ad6155e664a40a68f76bbc2a817c.pdf`
- **Keywords**: time series, diffusion models, frequency spectrum

### Abstract

Time series with missing or irregularly sampled data are a persistent challenge in machine learning. Many methods operate on the frequency-domain, relying on the Fast Fourier Transform (FFT) which assumes uniform sampling, therefore requiring prior interpolation that can distort the spectra. To address this limitation, we introduce a differentiable Lomb--Scargle layer that enables a reliable computation of the power spectrum of irregularly sampled data.
We integrate this layer into a novel score-based diffusion model (LSCD) for time series imputation conditioned on the entire signal spectrum. 
Experiments on synthetic and real-world benchmarks demonstrate that our method recovers missing data more accurately than purely time-domain baselines, while simultaneously producing consistent frequency estimates. Crucially, our method can be easily integrated into learning frameworks, enabling broader adoption of spectral guidance in machine learning approaches involving incomplete or irregular data.

### BibTeX

```
@inproceedings{
fons2025lscd,
title={{LSCD}: Lomb--Scargle Conditioned Diffusion for Time series Imputation},
author={Elizabeth Fons and Alejandro Sztrajman and Yousef El-Laham and Luciana Ferrer and Svitlana Vyetrenko and Manuela Veloso},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=GdYg0Ohx0k}
}
```

## UnHiPPO: Uncertainty-aware Initialization for State Space Models

- **Authors**: Marten Lienen, Abdullah Saydemir, Stephan Günnemann
- **Venue**: ICML 2025 poster
- **Primary area**: deep_learning->sequential_models_time_series
- **OpenReview**: `https://openreview.net/forum?id=U8GUmxnzXn`
- **PDF**: `https://openreview.net/pdf/20432296a22e025c99c7606bf8be6f3b437ebc58.pdf`
- **Keywords**: state space, uncertainty, hippo, mamba, kalman, noise, filter

### Abstract

State space models are emerging as a dominant model class for sequence problems with many relying on the HiPPO framework to initialize their dynamics. However, HiPPO fundamentally assumes data to be noise-free; an assumption often violated in practice. We extend the HiPPO theory with measurement noise and derive an uncertainty-aware initialization for state space model dynamics. In our analysis, we interpret HiPPO as a linear stochastic control problem where the data enters as a noise-free control signal. We then reformulate the problem so that the data become noisy outputs of a latent system and arrive at an alternative dynamics initialization that infers the posterior of this latent system from the data without increasing runtime. Our experiments show that our initialization improves the resistance of state-space models to noise both at training and inference time.

### BibTeX

```
@inproceedings{
lienen2025unhippo,
title={UnHi{PPO}: Uncertainty-aware Initialization for State Space Models},
author={Marten Lienen and Abdullah Saydemir and Stephan G{\"u}nnemann},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=U8GUmxnzXn}
}
```

## When Will It Fail?: Anomaly to Prompt for Forecasting Future Anomalies in Time Series

- **Authors**: Min-Yeong Park, Won-Jeong Lee, Seong Tae Kim, Gyeong-Moon Park
- **Venue**: ICML 2025 poster
- **Primary area**: deep_learning->sequential_models_time_series
- **OpenReview**: `https://openreview.net/forum?id=Dqp6IMI3gQ`
- **PDF**: `https://openreview.net/pdf/4a719cd370dbee97e8a4d985b1b58bfa27ba71a6.pdf`
- **Keywords**: Time series forecasting, time series anomaly detection

### Abstract

Recently, forecasting future abnormal events has emerged as an important scenario to tackle realworld necessities. However, the solution of predicting specific future time points when anomalies will occur, known as Anomaly Prediction (AP), remains under-explored. Existing methods dealing with time series data fail in AP, focusing only on immediate anomalies or failing to provide precise predictions for future anomalies. To address AP, we propose a novel framework called Anomaly to Prompt (A2P), comprised of Anomaly-Aware Forecasting (AAF) and Synthetic Anomaly Prompting (SAP). To enable the forecasting model to forecast abnormal time points, we adopt a strategy to learn the relationships of anomalies. For the robust detection of anomalies, our proposed SAP introduces a learnable Anomaly Prompt Pool (APP) that simulates diverse anomaly patterns using signal-adaptive prompt. Comprehensive experiments on multiple real-world datasets demonstrate the superiority of A2P over state-of-the-art methods, showcasing its ability to predict future anomalies.

### BibTeX

```
@inproceedings{
park2025when,
title={When Will It Fail?: Anomaly to Prompt for Forecasting Future Anomalies in Time Series},
author={Min-Yeong Park and Won-Jeong Lee and Seong Tae Kim and Gyeong-Moon Park},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=Dqp6IMI3gQ}
}
```
