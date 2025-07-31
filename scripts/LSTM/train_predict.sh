num_layers=2
lr=0.001
epochs=20
dropout=0.5
hidden_size=64
seq_length=30
test_ratio=0.2

# Predicting LSTM model with each news model and output
for model in "finbert" "roberta" "deberta"; do
    for target_column in "Float_Price" "Factor_Price" "Delta_Price" "Binary_Price"; do

        feature_columns="Close Volume total_news_count ${model}_label_negative_sum ${model}_label_positive_sum ${model}_label_neutral_sum ${model}_majority_vote ${model}_count_positive ${model}_count_negative ${model}_count_neutral"

        python stock_prediction/modeling/train.py \
            --model_name lstm_model \
            --data_path data/processed/AAPL_preprocessed_dataset_with_features.csv \
            --feature_columns $feature_columns \
            --target_column $target_column \
            --scaler_path models/AAPL_lstm_model_norm.npz \
            --seq_length $seq_length \
            --hidden_size $hidden_size \
            --num_layers $num_layers \
            --dropout $dropout \
            --test_ratio $test_ratio \
            --lr $lr \
            --epochs $epochs \
            --force_retrain

        
        python stock_prediction/modeling/train.py \
            --model_name lstm_model \
            --data_path data/processed/AMZN_preprocessed_dataset_with_features.csv \
            --feature_columns $feature_columns \
            --target_column $target_column \
            --scaler_path models/AMZN_lstm_model_norm.npz \
            --seq_length $seq_length \
            --hidden_size $hidden_size \
            --num_layers $num_layers \
            --dropout $dropout \
            --test_ratio 0.2 \
            --lr $lr \
            --epochs $epochs 

        python stock_prediction/modeling/train.py \
            --model_name lstm_model \
            --data_path data/processed/MSFT_preprocessed_dataset_with_features.csv \
            --feature_columns $feature_columns \
            --target_column $target_column \
            --scaler_path models/MSFT_lstm_model_norm.npz \
            --seq_length $seq_length \
            --hidden_size $hidden_size \
            --num_layers $num_layers \
            --dropout $dropout \
            --test_ratio $test_ratio \
            --lr $lr \
            --epochs $epochs 

        python stock_prediction/modeling/train.py \
            --model_name lstm_model \
            --data_path data/processed/TSLA_preprocessed_dataset_with_features.csv \
            --feature_columns $feature_columns \
            --target_column $target_column \
            --scaler_path models/TSLA_lstm_model_norm.npz \
            --seq_length $seq_length \
            --hidden_size $hidden_size \
            --num_layers $num_layers \
            --dropout 0.5 \
            --test_ratio $test_ratio \
            --lr $lr \
            --epochs $epochs 

        python stock_prediction/modeling/train.py \
            --model_name lstm_model \
            --data_path data/processed/NFLX_preprocessed_dataset_with_features.csv \
            --feature_columns $feature_columns \
            --target_column $target_column \
            --scaler_path models/NFLX_lstm_model_norm.npz \
            --seq_length $seq_length \
            --hidden_size $hidden_size \
            --num_layers $num_layers \
            --dropout $dropout \
            --test_ratio $test_ratio \
            --lr $lr \
            --epochs $epochs 
        

        python stock_prediction/modeling/predict.py \
            --model_name lstm_model \
            --model_path models/lstm_model.pth \
            --scaler_path models/AAPL_lstm_model_norm.npz \
            --data_path data/processed/AAPL_preprocessed_dataset_with_features.csv \
            --feature_columns $feature_columns \
            --target_column $target_column \
            --ticker AAPL \
            --seq_length $seq_length \
            --hidden_size $hidden_size \
            --num_layers $num_layers \
            --dropout $dropout \
            --test_ratio 0.2 \
            --news_model $model

        
        python stock_prediction/modeling/predict.py \
            --model_name lstm_model \
            --model_path models/lstm_model.pth \
            --scaler_path models/AMZN_lstm_model_norm.npz \
            --data_path data/processed/AMZN_preprocessed_dataset_with_features.csv \
            --feature_columns $feature_columns \
            --target_column $target_column \
            --ticker AMZN \
            --seq_length $seq_length \
            --hidden_size $hidden_size \
            --num_layers $num_layers \
            --dropout $dropout \
            --test_ratio $test_ratio \
            --news_model $model

        python stock_prediction/modeling/predict.py \
            --model_name lstm_model \
            --model_path models/lstm_model.pth \
            --scaler_path models/MSFT_lstm_model_norm.npz \
            --data_path data/processed/MSFT_preprocessed_dataset_with_features.csv \
            --feature_columns $feature_columns \
            --target_column $target_column \
            --ticker MSFT \
            --seq_length $seq_length \
            --hidden_size $hidden_size \
            --num_layers $num_layers \
            --dropout $dropout \
            --test_ratio $test_ratio \
            --news_model $model

        python stock_prediction/modeling/predict.py \
            --model_name lstm_model \
            --model_path models/lstm_model.pth \
            --scaler_path models/TSLA_lstm_model_norm.npz \
            --data_path data/processed/TSLA_preprocessed_dataset_with_features.csv \
            --feature_columns $feature_columns \
            --target_column $target_column \
            --ticker TSLA \
            --seq_length $seq_length \
            --hidden_size $hidden_size \
            --num_layers $num_layers \
            --dropout $dropout \
            --test_ratio $test_ratio \
            --news_model $model

        python stock_prediction/modeling/predict.py \
            --model_name lstm_model \
            --model_path models/lstm_model.pth \
            --scaler_path models/NFLX_lstm_model_norm.npz \
            --data_path data/processed/NFLX_preprocessed_dataset_with_features.csv \
            --feature_columns $feature_columns \
            --target_column $target_column \
            --ticker NFLX \
            --seq_length $seq_length \
            --hidden_size $hidden_size \
            --num_layers $num_layers \
            --dropout $dropout \
            --test_ratio $test_ratio \
            --news_model $model
        
        rm -f models/*.pth
        rm -f models/*.npz
    done
done