python absa_pipeline.py \
--task_name yelp-absa-dataset \
--target_extraction_model_dir ../BERT-Target-Extraction/dumped/Term_Extraction_Dedup_Noleak/best_model_dir \
--polarity_classification_model_dir ../BERT-ABSA-Polarity/dumped/ABSA_Polarity_Prediction_Dedup_Noleak/best_model_dir \
--data_path ../demo_data/yelp_nips17/data/sentiment.dev.0 \
--output_path dumped/yelp_sentiment.dev.0.txt

python absa_pipeline.py \
--task_name yelp-absa-dataset \
--target_extraction_model_dir ../BERT-Target-Extraction/dumped/Term_Extraction_Dedup_Noleak/best_model_dir \
--polarity_classification_model_dir ../BERT-ABSA-Polarity/dumped/ABSA_Polarity_Prediction_Dedup_Noleak/best_model_dir \
--data_path ../demo_data/yelp_nips17/data/sentiment.dev.1 \
--output_path dumped/yelp_sentiment.dev.1.txt
