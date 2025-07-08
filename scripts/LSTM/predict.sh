python stock_prediction/modeling/predict.py \
    --model_name lstm_model \
    --model_path models/lstm_model.pth \
    --scaler_path models/lstm_model_norm.npz \
    --predictions_path data/processed/test_predictions.csv \
    --seq_length 30 \
    --hidden_size 64 \
    --num_layers 2 \
    --dropout 0.2 \
    --test_ratio 0.2