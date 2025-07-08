python stock_prediction/modeling/train.py \
    --model_name lstm_model \
    --data_path data/processed/AMZN_FINBERT_FLOAT.csv \
    --scaler_path models/lstm_model_norm.npz \
    --seq_length 30 \
    --hidden_size 64 \
    --num_layers 2 \
    --dropout 0.2 \
    --test_ratio 0.2