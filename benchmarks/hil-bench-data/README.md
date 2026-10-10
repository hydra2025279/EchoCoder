---
dataset_info:
  features:
  - name: task_id
    dtype: string
  - name: task_type
    dtype: string
  - name: repo_or_db_name
    dtype: string
  - name: repo_or_db_download_link
    dtype: string
  - name: problem
    dtype: string
  - name: business_info
    list: string
  - name: schema_descriptions_download_links
    list: string
  - name: ground_truth_answer
    dtype: string
  - name: test_patch
    dtype: string
  - name: tests_to_pass
    list: string
  - name: test_files
    list: string
  - name: blocker_registry
    list:
    - name: description
      dtype: string
    - name: example_questions
      list: string
    - name: id
      dtype: string
    - name: resolution
      dtype: string
    - name: subtype
      dtype: string
    - name: type
      dtype: string
  - name: uid
    dtype: string
  splits:
  - name: train
    num_bytes: 3790772
    num_examples: 200
  download_size: 1645462
  dataset_size: 3790772
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
