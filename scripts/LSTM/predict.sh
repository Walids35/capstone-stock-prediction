target_column="Binary_Price"
news_model="svm"
feature_columns="Close Open High Low Volume total_news_count majority_vote_$news_model mean_score_$news_model"

python stock_prediction/modeling/predict.py \
    --model_name lstm_model \
    --model_path models/lstm_model.pth \
    --scaler_path models/lstm_model_norm.npz \
    --data_path data/processed/AAPL_preprocessed_dataset_with_features.csv \
    --feature_columns $feature_columns \
    --target_column $target_column \
    --ticker AAPL \
    --seq_length 30 \
    --hidden_size 64 \
    --num_layers 2 \
    --dropout 0.2 \
    --test_ratio 0.2 \
    --news_model $news_model

python stock_prediction/modeling/predict.py \
    --model_name lstm_model \
    --model_path models/lstm_model.pth \
    --scaler_path models/lstm_model_norm.npz \
    --data_path data/processed/AMZN_preprocessed_dataset_with_features.csv \
    --feature_columns $feature_columns \
    --target_column $target_column \
    --ticker AMZN \
    --seq_length 30 \
    --hidden_size 64 \
    --num_layers 2 \
    --dropout 0.2 \
    --test_ratio 0.2 \
    --news_model $news_model

python stock_prediction/modeling/predict.py \
    --model_name lstm_model \
    --model_path models/lstm_model.pth \
    --scaler_path models/lstm_model_norm.npz \
    --data_path data/processed/MSFT_preprocessed_dataset_with_features.csv \
    --feature_columns $feature_columns \
    --target_column $target_column \
    --ticker MSFT \
    --seq_length 30 \
    --hidden_size 64 \
    --num_layers 2 \
    --dropout 0.2 \
    --test_ratio 0.2 \
    --news_model $news_model

python stock_prediction/modeling/predict.py \
    --model_name lstm_model \
    --model_path models/lstm_model.pth \
    --scaler_path models/lstm_model_norm.npz \
    --data_path data/processed/TSLA_preprocessed_dataset_with_features.csv \
    --feature_columns $feature_columns \
    --target_column $target_column \
    --ticker TSLA \
    --seq_length 30 \
    --hidden_size 64 \
    --num_layers 2 \
    --dropout 0.2 \
    --test_ratio 0.2 \
    --news_model $news_model

python stock_prediction/modeling/predict.py \
    --model_name lstm_model \
    --model_path models/lstm_model.pth \
    --scaler_path models/lstm_model_norm.npz \
    --data_path data/processed/NFLX_preprocessed_dataset_with_features.csv \
    --feature_columns $feature_columns \
    --target_column $target_column \
    --ticker NFLX \
    --seq_length 30 \
    --hidden_size 64 \
    --num_layers 2 \
    --dropout 0.2 \
    --test_ratio 0.2 \
    --news_model $news_model