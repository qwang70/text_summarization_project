# text_summarization_project
Author: Qiwen Wang, Zhixiu Liu, Yinchen Xu

-----
Blind and visually impaired (BVI) people often struggle to find out information on products when shopping online. 
Online reviews provide a promising source, yet they remain infeasible for BVI users to search through an extensive and often-duplicated amount of text. 

In this repo, we employed both LSTM and bi-directional LSTM with self-attention layers and applied the beam search strategy to decode the sum- maries. Our results are comprehensive and natural enough, and we also conducted evaluations and comparisons between these different approaches.

## Repository Structure

- baseline: Implement our baseline using k-means. 
  - k_means: Run the script using `python3 k_means`. It outputs the top k summaries closest to the input reviews.
- LSTM: Encoder-decoder based models.
  - lstm_train_on_review_title.py: Single direction LSTM model.
  - bilstm_train_on_review_title.py: Bi-directional LSTM model.
  - tune_param.sh: Script to tune the embedding size.
  - calculate_metrics.py: Script to compute the ROUGE-1 and ROUGE-2 score.
- Extraction and abstrastion based models: Refer to our [other repo](https://github.com/qwang70/PreSumm) for the code and the output.

```
usage: lstm_train_on_review_title.py [-h] [--epoch EPOCH]
                                    [--embedding EMBEDDING] [--latent LATENT]
                                    [--max-text-len MAX_TEXT_LEN]
                                    [--max-summary-len MAX_SUMMARY_LEN]

```
