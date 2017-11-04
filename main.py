from trainer import Trainer
from resnet import resnet18
import pandas as pd


model = resnet18()

trainer = Trainer(model,
        lr=.001,
        weight_decay=0.001,
        epochs=1000,
        step_size=200,
        gamma=0.1,
        batch_size=256, 
        use_cuda=False, 
        gpu_idx=0)
        
# train  
train_df = pd.read_csv('Data/processed_train.csv')       
trainer.train(train_df)