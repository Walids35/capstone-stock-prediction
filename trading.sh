#!/bin/bash


for model in 'finbert' 'roberta' 'deberta' 'lr' 'rf' 'svm'; do
    for ticker in 'AMZN' 'AAPL' 'TSLA' 'NFLX' 'MSFT'; do
        echo "Running trading simulation for ${ticker} with ${model} model"

        case $model in
            "deberta")
                feature_columns="Close Volume total_news_count deberta_majority_vote deberta_count_positive deberta_count_negative deberta_count_neutral deberta_label_positive_sum deberta_label_negative_sum deberta_label_neutral_sum"
                ;;
            "finbert")
                feature_columns="Close Volume total_news_count finbert_majority_vote finbert_count_positive finbert_count_negative finbert_count_neutral finbert_label_positive_sum finbert_label_negative_sum finbert_label_neutral_sum"
                ;;
            "lr")
                feature_columns="Close Volume total_news_count lr_majority_vote lr_count_positive lr_count_negative lr_count_neutral"
                ;;
            "rf")
                feature_columns="Close Volume total_news_count rf_majority_vote rf_count_positive rf_count_negative rf_count_neutral"
                ;;
            "roberta")
                feature_columns="Close Volume total_news_count roberta_majority_vote roberta_count_positive roberta_count_negative roberta_count_neutral roberta_label_positive_sum roberta_label_negative_sum roberta_label_neutral_sum"
                ;;
            "svm")
                feature_columns="Close Volume total_news_count svm_majority_vote svm_count_positive svm_count_negative svm_count_neutral"
                ;;
        esac


        python stock_prediction/trading_simulation.py \
            --data_path data/processed/${ticker}_preprocessed_dataset_with_features.csv \
            --seq_length 30 \
            --test_ratio 0.2 \
            --feature_columns $feature_columns \
            --target_column Binary_Price \
            --scaler_path models/${ticker}_lstm_Binary_Price_${model}_scaler.pkl \
            --model_path models/lstm_Binary_Price_${model}_model.pth \
            --ticker ${ticker} \
            --news_model ${model}
    done
done
