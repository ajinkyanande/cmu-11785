To run the entire pipeline: python hw4p2.py

Experimentation:

    01) Epochs : 50 : total number of epochs trained with varying teacher forcing rate

    02) Batch Size : 128 : maximum that fits the hardware

    03) Weight Decay : 5e-6 : standard LAS value

    04) Learning Rate Scheduler : cosine annealing : starts with 1e-3 and is decayed till is 1e-5

    05) Teacher Forcing Rate Scheduler : teacher forcing rate was manually set to values [1.0, 0.75, 0.6, 0.3, 0.1] according th model convergence

    06) Network : LAS(
                     (encoder): Listener(
                         (base_lstm): LSTM(15, 512, batch_first=True, bidirectional=True)
                         (ld1): LockedDropout(p=0.4)
                         (pBLSTM1): pBLSTM(
                         (blstm): LSTM(2048, 512, batch_first=True, bidirectional=True)
                         )
                         (ld2): LockedDropout(p=0.3)
                         (pBLSTM2): pBLSTM(
                         (blstm): LSTM(2048, 512, batch_first=True, bidirectional=True)
                         )
                         (ld3): LockedDropout(p=0.3)
                         (pBLSTM3): pBLSTM(
                         (blstm): LSTM(2048, 512, batch_first=True, bidirectional=True)
                         )
                     )
                     (decoder): Speller(
                         (attention): Attention(
                         (key_projection): Linear(in_features=1024, out_features=256, bias=True)
                         (value_projection): Linear(in_features=1024, out_features=256, bias=True)
                         (query_projection): Linear(in_features=512, out_features=256, bias=True)
                         (context_projection): Linear(in_features=256, out_features=256, bias=True)
                         (softmax): Softmax(dim=1)
                         )
                         (embedding): Embedding(30, 512, padding_idx=29)
                         (lstm_cells): Sequential(
                         (0): LSTMCell(768, 512)
                         (1): LSTMCell(512, 512)
                         )
                         (char_prob): Linear(in_features=768, out_features=30, bias=True)
                     )
                     )

    07) Network Weights Initialization : LSTM weights were initialized with uniform distribution between -0.1 and 0.1 as given in LAS paper

    08) WandB Runs : wandb project is made public and can be found here : https://wandb.ai/ajinkyanande111/hw4p2
