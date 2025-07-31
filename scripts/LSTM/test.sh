feature_columns="Close Volume count original_sentiment_majority sentiment_score_mean"
target_column="TargetBinary"
news_model="test"

python stock_prediction/modeling/train.py \
    --model_name lstm_model \
    --data_path data/external/amazon_merged.csv \
    --feature_columns $feature_columns \
    --target_column $target_column \
    --scaler_path models/lstm_model_norm.npz \
    --seq_length 30 \
    --hidden_size 64 \
    --num_layers 2 \
    --dropout 0.2 \
    --test_ratio 0.2

python stock_prediction/modeling/predict.py \
    --model_name lstm_model \
    --model_path models/lstm_model.pth \
    --scaler_path models/lstm_model_norm.npz \
    --data_path data/external/amazon_merged.csv \
    --feature_columns $feature_columns \
    --target_column $target_column \
    --ticker test \
    --seq_length 30 \
    --hidden_size 64 \
    --num_layers 2 \
    --dropout 0.2 \
    --test_ratio 0.2 \
    --news_model $news_model