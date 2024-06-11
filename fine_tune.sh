#/usr/bin bash

unionai run --copy-all --remote \
    llm_edge_finetuning/workflows_news.py train_workflow \
    --config config/phi_3_mini_128k_instruct.json \
    --categories '["science", "technology", "business"]'
