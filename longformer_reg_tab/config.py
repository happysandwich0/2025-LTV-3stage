HP = dict(
    max_len=128,
    batch_size=64,
    epochs=1,
    patience=2,
    d_model=32,
    nhead=1,
    nlayers=2,
    lr=2e-4,
    wd=2e-4,
    y_col='PAY_AMT',
    min_freq=1,
    top_k_vocab=20,
    num_workers=8,
    base_rate=0.03675,
    
    transformation_mode='log1p', 
    loss_mode='mae',      
    huber_delta=1.0         
)
