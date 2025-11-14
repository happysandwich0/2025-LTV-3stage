# Pipeline Architecture

```mermaid
flowchart LR
    %% Direction
    A[Raw Player Logs<br/>First 5 Days After Launch] --> B[Feature Engineering<br/>Aggregation]

    %% Stage 1
    B --> S1[Stage 1 Payer Classification<br/>CatBoost LightGBM XGBoost]
    S1 -->|Non Payer 0| NP[Non Payer Segment<br/>LTV Baseline]
    S1 -->|Payer 1| S2[Stage 2 High Value Classification<br/>CatBoost LightGBM TabPFNClassifier]

    %% Stage 2
    S2 -->|Low Spender| R_low[Stage 3B Low Value Regressor<br/>LightGBM XGBoost TabPFNRegressor]
    S2 -->|High Spender Whale| R_high[Stage 3A High Value Regressor]

    %% Final Output
    R_low --> OUT[Final LTV Prediction]
    R_high --> OUT

    %% Optimization Notes
    subgraph STAGE1[Stage 1 Tuning]
        T1[Optuna PR AUC Tuning<br/>Cutoff Strategy Prior vs Reweight<br/>Selected By F1 Maximization]
    end

    subgraph STAGE2[Stage 2 Tuning]
        T2[Optuna PR AUC and F Beta Tuning<br/>Recall Focus For High Value Users]
    end

    subgraph STAGE3[Stage 3 Tuning]
        T3[Optuna MAE Minimization<br/>Separate Heads For High and Low Segments]
    end

    S1 -. tuning .-> T1
    S2 -. tuning .-> T2
    R_low -. tuning .-> T3
    R_high -. tuning .-> T3
