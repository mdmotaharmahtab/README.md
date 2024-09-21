# Machine Learning Engineer
#### ML Libraries
`PyTorch, PyTorch Lightning, Huggingface, LangChain, OpenCV, Flair, OpenNMT, AllenNLP, NLTK, Pandas, Matplotlib, NumPy`
#### Frameworks:
`NLTK, PyTorch, Tensorflow, PyTorch Lightning, Huggingface, OpenCV, OpenMMLab, Keras, FastAI, Ray Tune, Wandb, TensorBoard, Flair, OpenNMT, AllenNLP, Pandas, Matplotlib, Seaborn, Django, FastAPI`
#### Web Frameworks
`Flask, Django, FastAPI, Streamlit`
#### Developer Tools
`Git, Docker, Locust, pre-commit`
#### ML Tools
`Triton, Dask, DVC, MLflow, Elasticsearch, Qdrant, Ray Tune, Wandb, TensorBoard, Pytorch Profiler`
#### Programming
`Python, Bash, SQL`

## Education
- B.Sc (Hons), Computer Science & Engineering | BRAC University (_April 2022_)	| CGPA: 3.99							       		
- H.S.C, Science	| Notredame College (_2017_) | GPA 5.0	 			        		
- S.S.C, Science | A.K School and College (_2015_) | GPA 5.0

## Work Experience
**Jr. Machine Learning Engineer @ GIGATECH, BEXIMCO (_September 2022 - Present_)**
  - Created new state-of-the-art systems for a plethora of Bangla NLP tasks e.g. named entity recognition (NER), Parts of Speech (POS), Lemmatization, and Emotion recognition. Bangla Lemmatization and Emotion recognition systems are publicly available at [here](https://github.com/eblict-gigatech/BanLemma) and [here](https://sentiment.bangla.gov.bd/) respectively.
  - For efficient deployment of ML services which use Large Language Models (LLM), used Optimum and Onnx for onnx conversion and converted into Nvidia TensorRT(TRT) format for further optimization. Used Locust for load testing and pytorch profiler to reduce bottlenecks. Used Nvidia Triton Inference Server (TIS) as the inference server to facilitate concurrent request serving and scheduling, batch inference and response caching.
  - Created REST APIs using FastAPI for hosting ML inference endpoints. Used MongoDB for response caching in NVIDIA Triton.
  - Created data augmentation pipeline to handle the class imbalance problem in sequence tagging tasks. Formulated a general test set creation guidelines for unbiased classification performance calculation.}
  - Used Qdrant vector DB for fast semantic searching, Dask to analyze and query big dataframes, DVC for dataset versioning and MLflow for model, artifact and experiment versioning.
  - Created pipeline for Natural Language generation (NLG) in Bangla for both encoder models like BERT and auto-regressive models like GPT2. Analyzed and overcame common issues like repetitive text generation, and unmeaningful word generation in NLG for Bangla.    

**Research Assistant (Remote) @ Qatar Computing Research Institute (QCRI) (_Septembert 2021 - December 2021_)**
- Pretrained a HuBERT model on Bangla ASR dataset for joint task of speech and speaker recognition pipeline using SpeechBrain.
- Assisted in enriching existing open source Bangla ASR datasets by adding more scirpted audion and correcting existing annotation.

**Teacher Assistant & BRAC University (_April 2020 - April 2022_)**
- Helped students with different coding assignments and helped teachers in checking scripts
- Assisted students in conducting research in various fields and submitting papers into conference.
- Assisted Teachers in lab classes and helped students with different course materials during consultation hour.

## Projects
### Bangla Clickbait Detector App
[repo](https://github.com/mdmotaharmahtab/Bangla-Clickbait-Detector-App)
-  Demo app created as a part of research work on Bangla Clickbait Detection using GAN‑Transformers. It takes a Bangla article title as input and
outputs whether the title is a clickbait or non‑clickbait along with the prediction probability score. GAN‑Transformers is a Transformer network
trained in generative adversarial training framework.
- Tools used: Pytorch, Streamlit, Node.js

![clickbait_detection_demo](/assets/img/clickbait_detection_demo.gif)

### Bangla Article Headline Categorizer App
[repo](https://github.com/mdmotaharmahtab/Bangla-Headline-Categorizer-App)

- Can categorize Bangla article headlines into eight different categories ‑ Economy, Education, Entertainment, Politics, International, Sports,
National, and Science & Technology
- Models used: State‑of‑the‑art Bangla ELECTRA model, Dataset used: Patrika Dataset. ‑ contains 400𝑘 Bangla news articles from prominent
Bangla news sites.
- Tools used: Pytorch, Streamlit, Node.js
![Headline Categorizer Demo](/assets/img/headline_detector_demo.gif)

### EBRAC ‑ Online Learning App
[repo](https://github.com/mdmotaharmahtab/EBRAC)

- A comprehensive online education platform where instructors can create different courses, upload course content, enrol students, see students’
marks, prepare questions, take quizzes etc.
- Students can enrol in courses, view course contents, participate in exams and see results
- Tools used: Django, Bootstrap, Node.js
![EBRAC Learning App Demo](/assets/img/Functionalities.gif)

### Veggie - Vegetarian Recipe Maker App
[repo](https://github.com/mdmotaharmahtab/django_vegetarian_recipe)

- This web app allows users to view different vegetarian recipes, see their total calories, nutrients like protein, carbohydrate, fat and their ingre‑
dients.
- Users can create their own vegetarian recipes by mixing different ingredients available on the web app. They can also see the total nutrients
and calories of their created recipe
- Tools used: Django, Bootstrap, Node.js
![Veggy recipe maker app Demo](/assets/img/Functionalities_recipe.gif)

## Notable Publications
### BanglaBait: A Bangla Clickbait Detection Dataset for Identifying Clickbaits in Bangla News Articles ‑ [source](https://github.com/mdmotaharmahtab/BanglaBait)
<em>19th International Conference Recent Advances in Natural Language Processing (RANLP) (Varna (Bulgaria),2023)</em>
- First Bangla Clickbait News Article Dataset containing 15,056 data instances, each containing article title, content, clickbait/non clickbait label, article source, article category, article publish‑time, translated English title and content.
- Investigated with various semi‑supervised learning methods and compared it with supervised learning methods to prove the former’s superiority.

### BanLemma: A Word Formation Dependent Rule and Dictionary Based Bangla Lemmatizer ‑ [source](https://github.com/eblict-gigatech/BanLemma)
<em>The 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP) (Singapore, 2023)</em>
- State-of-the-art Bangla Rule Based Lemmatizer which proposes a novel iterative suffix stripping approach based on the part-of-speech tag of a word.
- To create lemmatization rules, an extensive Bangla text corpus of 90.65M unique sentences are analyzed.
- Shows superior performance than all previously published Bangla lemmatization methods on existing datasets.

### A GAN‑BERT Based Approach for Bengali Text Classification with a Few Labeled Examples - [source](https://link.springer.com/chapter/10.1007/978-3-031-20859-1_3)
<em>19th International Conference on Distributed Computing and Artificial Intelligence (DCAI) (Guimarães (Portugal),2022)</em>
- Trained state‑of‑the‑art Transformer networks in an adversarial fashion using Generative Adversarial Network (GAN) to achieve superior performance when labeled dataset size is too small.
- First Bangla Paper to investigate the application of GAN‑BERT on Bangla text classification tasks

### Comparative Analysis on Joint Modeling of Emotion and Abuse Detection in Bangla language - [source](https://link.springer.com/chapter/10.1007/978-3-031-12641-3_17)
<em>5th International Conference on Advances in Computing and Data Sciences (ICACDS) (GPCET, Kurnool, India, 2022)</em>
- A comparative analysis of different researches made on detecting emotional and abusive Bangla language
- Present the best approach that tailors certain attributes of emotional and abusive language detection with respect to their prognosis performance and their implementation toughness in Bangla lingo.

## Awards
- Winner, BRACU Intra University Programming Contest 2019
- Merit Scholarship Award, BRAC University  (2020-2022)

## Articles
### Sparse Transformers Explained | Part 1 | Medium
- Explained Sparse Transformers whcih reduces the computation complexity of the Transformer networks. GPT-3 uses the Sparse Transformers architecture in their Transformers.
- [URL](https://medium.com/@mahtab27672767/sparse-transformers-explained-part-1-aacbe10dca4a)

## Open Source Contributions
### Flair | [PR Link](https://github.com/flairNLP/flair/pull/3449)
- Flair is a framework for state-of-the-art NLP embeddings and training sequence models. Contributed to fixing a bug in the Flair framework which was causing incorrect prediction distribution output for a sequence of tokens in sequence classification tasks (Chosen to be merged in their next version release.)

## Contact
* Email: md.motahar.mahtab@g.bracu.ac.bd
* LinkedIn: https://www.linkedin.com/in/motahar-mahtab/
* [Resume](https://drive.google.com/file/d/1GuCoGKKE27Jna7_jggvio2viTC_wDeKg/view?usp=sharing)
