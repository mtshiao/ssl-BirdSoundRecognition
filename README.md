Bird Sound Classifier Using Self-Supervised Learning
This repository provides the code and resources for our bird sound classifier, which leverages self-supervised learning to analyze vast amounts of unlabeled soundscape recordings collected from subtropical montane forests. The model is pre-trained on recordings captured during the first hour after sunrise and fine-tuned for 31 bird species native to Taiwan's montane regions. Key features of the model include:

Focused on dawn chorus bird song recognition: The model is designed to classify bird songs during the dawn chorus. We trained the model using recordings from this specific period and validated its effectiveness through practical inference tests.
Addressing data imbalances and cross-domain challenges: By incorporating a small portion of open-source datasets and employing augmentation techniques, the model effectively handles data imbalance and domain variation issues.
Enhanced robustness with a 'NOTA' category: To improve adaptability in open-set recognition tasks, a "None of the Above" (NOTA) category is introduced, enabling the model to better handle non-target sounds and background noise.
This classifier is tailored for ecological studies and supports bird monitoring in remote subtropical montane ecosystems. For detailed information, please refer to our publication in Ecological Information.

Reference
Wei, Y.C., Chen, W.L., Tuanmu, M.L., Lu, S.S., Shiao, M.T., 2025. Advanced montane bird monitoring using self-supervised learning and transformer on passive acoustic data. Ecological Information （under review).
