# %% [code] {"jupyter":{"outputs_hidden":false}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"execution":{"iopub.status.busy":"2025-01-29T05:37:56.408689Z","iopub.execute_input":"2025-01-29T05:37:56.408970Z","iopub.status.idle":"2025-01-29T05:37:56.419446Z","shell.execute_reply.started":"2025-01-29T05:37:56.408946Z","shell.execute_reply":"2025-01-29T05:37:56.418630Z"},"jupyter":{"outputs_hidden":false}}
import os
os.listdir('/kaggle/input/readability-data')

# %% [code] {"execution":{"iopub.status.busy":"2025-01-29T05:37:58.064388Z","iopub.execute_input":"2025-01-29T05:37:58.064735Z","iopub.status.idle":"2025-01-29T05:38:06.064996Z","shell.execute_reply.started":"2025-01-29T05:37:58.064705Z","shell.execute_reply":"2025-01-29T05:38:06.064306Z"},"jupyter":{"outputs_hidden":false}}
from transformers import AutoConfig, AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
import torch.optim as optim
import numpy as np
import pandas as pd
import torch.nn as nn
import random
import torch 
import time
import math
import os
import gc

# %% [code] {"execution":{"iopub.status.busy":"2025-01-29T05:38:06.066023Z","iopub.execute_input":"2025-01-29T05:38:06.066454Z","iopub.status.idle":"2025-01-29T05:38:06.071315Z","shell.execute_reply.started":"2025-01-29T05:38:06.066431Z","shell.execute_reply":"2025-01-29T05:38:06.070474Z"},"jupyter":{"outputs_hidden":false}}
def eval_mse(model, val_dataloader:DataLoader, device):
    model.eval()

    total_loss, total_samples = 0,0

    with torch.no_grad():
        for (input_ids, attention_mask, target) in val_dataloader:
            
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            target = target.to(device=device)
            output = model(input_ids, attention_mask)

            loss = nn.MSELoss(reduction='sum')(output.flatten(), target)
            total_loss += loss.item()
            total_samples += target.size(0)
    
    return total_loss / total_samples

# %% [code] {"execution":{"iopub.status.busy":"2025-01-29T05:38:06.072783Z","iopub.execute_input":"2025-01-29T05:38:06.073042Z","iopub.status.idle":"2025-01-29T05:38:06.093069Z","shell.execute_reply.started":"2025-01-29T05:38:06.073021Z","shell.execute_reply":"2025-01-29T05:38:06.092453Z"},"jupyter":{"outputs_hidden":false}}
# setting the seed value
def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    torch.backends.cudnn.deterministic = True

# %% [code] {"execution":{"iopub.status.busy":"2025-01-29T05:38:06.094229Z","iopub.execute_input":"2025-01-29T05:38:06.094505Z","iopub.status.idle":"2025-01-29T05:38:08.731293Z","shell.execute_reply.started":"2025-01-29T05:38:06.094476Z","shell.execute_reply":"2025-01-29T05:38:08.730646Z"},"jupyter":{"outputs_hidden":false}}
EVAL_SCHEDULE = [(0.50, 16), (0.49, 8), (0.48, 4), (0.47, 2), (-1., 1)]

train_df = pd.read_csv('/kaggle/input/readability-data/train.csv')
# remove incomplete entries 
train_df.drop(train_df[(train_df.target == 0) & 
                        (train_df.standard_error == 0)].index, inplace=True)

train_df.reset_index(drop=True, inplace=True)

test_df = pd.read_csv('/kaggle/input/readability-data/test.csv')
submission_df = pd.read_csv('/kaggle/input/readability-data/sample_submission.csv')

tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
MAX_LEN = 248
n_epochs = 5
BATCH_SIZE = 16

# %% [code] {"execution":{"iopub.status.busy":"2025-01-29T05:38:08.732097Z","iopub.execute_input":"2025-01-29T05:38:08.732364Z","iopub.status.idle":"2025-01-29T05:38:08.737931Z","shell.execute_reply.started":"2025-01-29T05:38:08.732339Z","shell.execute_reply":"2025-01-29T05:38:08.737100Z"},"jupyter":{"outputs_hidden":false}}

# build the dataset layer 
class LitDataset(Dataset):
    def __init__(self,df, inference_only=False):
        self.df = df
        self.inference_only = inference_only
        self.text = df.excerpt.tolist()

        if not self.inference_only:
            self.target = torch.tensor(df.target.values, dtype=torch.float32)
        
        self.encoded = tokenizer.batch_encode_plus(
            self.text,
            padding='max_length',
            max_length=MAX_LEN,
            truncation=True,
            return_attention_mask=True
        )

    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.encoded['input_ids'][idx])
        attention_mask = torch.tensor(self.encoded['attention_mask'][idx])

        if self.inference_only:
            return (input_ids, attention_mask)
        else:
            target = self.target[idx]
            return (input_ids, attention_mask, target)

# %% [code] {"execution":{"iopub.status.busy":"2025-01-29T05:38:08.738733Z","iopub.execute_input":"2025-01-29T05:38:08.739021Z","iopub.status.idle":"2025-01-29T05:38:08.751505Z","shell.execute_reply.started":"2025-01-29T05:38:08.738991Z","shell.execute_reply":"2025-01-29T05:38:08.750749Z"},"jupyter":{"outputs_hidden":false}}
# Model 
class LitModel(nn.Module):

    def __init__(self):
        super().__init__()

        config = AutoConfig.from_pretrained("FacebookAI/roberta-base")
        config.update({
            'output_hidden_states': True,
            'hdden_dropout_prob': 0.0,
            'layer_norm_eps': 1e-7
        })
        
        self.model = AutoModel.from_pretrained(
            "FacebookAI/roberta-base", config = config)
        
        self.attention = nn.Sequential(
            nn.Linear(768, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )

        self.regressor = nn.Sequential(
            nn.Linear(768, 1)
        )

    def forward(self,input_ids, attention_mask):
        output = self.model(input_ids, attention_mask)
        last_hidden_states = output[2][-1] # hidden states
        weights = self.attention(last_hidden_states)
        context_vector = torch.sum(weights * last_hidden_states, dim=1)
        return self.regressor(context_vector)

# %% [code] {"execution":{"iopub.status.busy":"2025-01-29T07:17:23.367098Z","iopub.execute_input":"2025-01-29T07:17:23.367444Z","iopub.status.idle":"2025-01-29T07:17:23.376093Z","shell.execute_reply.started":"2025-01-29T07:17:23.367414Z","shell.execute_reply":"2025-01-29T07:17:23.375176Z"},"jupyter":{"outputs_hidden":false}}

def train(model, optimizer, train_loader, val_dataloader, model_path, 
          scheduler = None, device = 'cpu', min_delta=1e-4, patience = 10):
    
    
    last_eval_step, step = 0,0
    best_val_rmse = float('inf')
    eval_period = EVAL_SCHEDULE[0][1]
    early_stop_all = False

    patience_counter = 0
    best_epoch = 0 
    
    start = time.time()
    improvement = None

    for epoch in range(n_epochs):
        model.train()
        val_rmse = None

        for batch_num, (input_ids, attention_mask, target) in enumerate(train_loader):    
            
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            target = target.to(device)
            
            optimizer.zero_grad() # nullify the gradients 

            output = model(input_ids=input_ids, attention_mask=attention_mask)

            loss = nn.MSELoss(reduction='mean')(output.flatten(), target)
            loss.backward()

            optimizer.step()

            if scheduler:
                scheduler.step()

            if batch_num >= last_eval_step + eval_period:
                
                elapsed_seconds = time.time() - start 
                num_steps = step - last_eval_step
                print(f'{num_steps} steps took {elapsed_seconds:0.3} seconds')
                last_eval_step = step 
                val_rmse = math.sqrt(eval_mse(model=model, val_dataloader=val_dataloader,device=device))

                print(f'Epoch {epoch} batch_num: {batch_num} val_rmse:{val_rmse}')
                for rmse, period in EVAL_SCHEDULE:
                    if val_rmse >= rmse:
                        last_eval_step = period
                        break 
                
                if val_rmse < best_val_rmse - min_delta:
                    patience_counter  = 0
                    best_val_rmse = val_rmse
                    best_epoch = epoch 
                    torch.save(model.state_dict(),model_path)
                    print(f'New best val rmse: {best_val_rmse}')
                else:
                    patience_counter += 1
                
                start = time.time()
                
                if patience_counter > patience:
                    early_stop_all = True
                    break

            step += 1
            
        if early_stop_all:
            print(f'No improvement in model scores. ')
            break
                    
    return best_val_rmse

# %% [code] {"execution":{"iopub.status.busy":"2025-01-29T05:38:49.514443Z","iopub.execute_input":"2025-01-29T05:38:49.514792Z","iopub.status.idle":"2025-01-29T05:38:49.520717Z","shell.execute_reply.started":"2025-01-29T05:38:49.514764Z","shell.execute_reply":"2025-01-29T05:38:49.519793Z"},"jupyter":{"outputs_hidden":false}}

def create_optimizer(model: torch.Tensor):
    named_parameters = list(model.named_parameters())

    roberta_parameters = named_parameters[:197]
    attention_parameters = named_parameters[199:203]
    regressor_parameters = named_parameters[203:]

    attention_group = [params for (name, params) in attention_parameters]
    regressor_group = [params for (name, params) in regressor_parameters]

    parameters = []
    parameters.append({'params':attention_group})
    parameters.append({'params':regressor_group})


    for layer_num, (name, params) in enumerate(roberta_parameters):
        weight_decay = 0.0 if 'bias' in name else 0.01

        lr = 2e-5

        if layer_num >= 69:
            lr = 5e-5
        
        if layer_num >= 133:
            lr = 1e-4

        parameters.append({
            'params': params,
            'lr': lr,
            'weight_decay':weight_decay
        })
    
    return optim.Adam(params=parameters)

# %% [code] {"execution":{"iopub.status.busy":"2025-01-29T07:17:46.818215Z","iopub.execute_input":"2025-01-29T07:17:46.818515Z","iopub.status.idle":"2025-01-29T07:49:35.960464Z","shell.execute_reply.started":"2025-01-29T07:17:46.818490Z","shell.execute_reply":"2025-01-29T07:49:35.959714Z"},"jupyter":{"outputs_hidden":false}}

if __name__ == '__main__':

    # make use of kfold 
    seed = 1000
    n_folds = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    list_val_rmse = []
    kfold = KFold(n_splits=n_folds, random_state=seed, shuffle=True)

    fold_results = []
    
    for fold, (train_indx, val_indx) in enumerate(kfold.split(train_df)):

        model_path = f'model_{fold + 1}.pth'

        set_random_seed(seed)

        train_dataset = LitDataset(df=train_df.loc[train_indx])
        valid_dataset = LitDataset(df=train_df.loc[val_indx])

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True, num_workers=2)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, drop_last=False, shuffle=False, num_workers=2)

        set_random_seed(seed + fold)

        model = LitModel().to(device)
        optimizer = create_optimizer(model=model)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=50,
            num_training_steps=n_epochs * len(train_loader)
        )

        rmse = train(model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_dataloader=valid_loader,
            model_path=model_path,
            scheduler=scheduler,
            device=device
            )
        
        fold_results.append({
            'fold': fold + 1,
            'model_path': model_path,
            'val_rmse': rmse
        })

        list_val_rmse.append(rmse)
        del model 
        gc.collect()
        torch.cuda.empty_cache()

    # Analyxe the results 
    mean_rmse = np.mean([fold['val_rmse'] for fold in fold_results])
    std_rmse = np.std([fold['val_rmse'] for fold in fold_results])
    best_fold = min(fold_results, key=lambda x:x['val_rmse'])

    print('Cross Validation Results')
    print(f'Mean Error: {mean_rmse:.4f} ± {std_rmse:.4f}')
    print(f'Best model from the fold {best_fold["fold"]} with rmse {best_fold["val_rmse"]}')
    
    print('Performance estimates')
    print(list_val_rmse)
    print(f'Mean: {np.array(list_val_rmse).mean()}')

# %% [code] {"jupyter":{"outputs_hidden":false}}
