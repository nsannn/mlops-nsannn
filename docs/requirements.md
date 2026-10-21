# Tweet Emotion Detection — Requirements v1.0

Date: 2026-10-21 | Author: Nojus Sandovs

## 1.1 Problem description
Create a API which can help social-media support team identify tweets and the emotions they encourage. 6 emotions will be tracked and identified (sadness, joy, love, anger, fear, surprise), build MLOps platform around it. Target: accuracy ≥ 0.85 and macro F1 ≥ 0.75 on held-out tweets,
with every experiment reproducible from Git + DVC + MLflow.


## 1.2 Data and model

- Data: [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) @ `cab853a`, seeded 10k stratified
  sample; licensed for educational and research use only.
- Model: [`google/bert_uncased_L-4_H-256_A-4`](https://huggingface.co/google/bert_uncased_L-4_H-256_A-4) @ `387825c`
  (11M parameters, Apache-2.0), fine-tuned on CPU.

## 2. Functional requirements

| ID    | Requirement                                                                 | Priority |
|-------|-----------------------------------------------------------------------------|----------|
| FR-01 | Classify a tweet into one of 6 emotions and return the label name and probability | Must |
| FR-02 | `dvc repro` builds the model from pinned raw data: prepare → train → evaluate | Must |
| FR-03 | Every training run logs params, metrics, dataset, `git_commit` and `dvc_raw_md5` tags and the model to the cluster MLflow | Must |
| FR-04 | Training runs from a container image (and as a Kubernetes Job) | Should |

## 3. Non-functional requirements

| ID     | Category    | Requirement                                   | KPI                              |
|--------|-------------|-----------------------------------------------|----------------------------------|
| NFR-01 | Quality     | Test accuracy                                 | ≥ 0.85                           |
| NFR-02 | Quality     | Macro F1 over 6 classes                       | ≥ 0.75                           |
| NFR-03 | Fairness    | Recall of the rarest class (surprise, ~4 %)   | ≥ 0.50                           |
| NFR-04 | Performance | Training run on CPU (4 threads, 8k tweets)    | ≤ 60 s                           |
| NFR-05 | Performance | Single-tweet inference latency (U2)           | p95 ≤ 50 ms on CPU               |
| NFR-06 | Size        | Saved model weights                           | ≤ 60 MB                          |
| NFR-07 | Security    | Secrets committed to Git                      | 0                                |