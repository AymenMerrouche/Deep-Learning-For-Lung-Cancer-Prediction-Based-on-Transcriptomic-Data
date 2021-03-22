from data_utils import *
from utils import *
from train import *
from models import *
from transformer_models import *
import datetime
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.distributions import Categorical


# Load args

# data paths args
with open('./configs/data_paths.yaml', 'r') as stream:
    data_paths_args  = yaml.load(stream,Loader=yaml.Loader)
    
    
# basic cnn args
with open('./configs/basic_cnn.yaml', 'r') as stream:
    basic_cnn_args  = yaml.load(stream,Loader=yaml.Loader)
    
# load the data  

net = Basic_CNN().to(device)

    
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=gen_args['lr'])

# Logging + Experiment

ignore_keys = {'no_tensorboard'}
# get hyperparameters with values in a dict
hparams = {**basic_cnn_args, **basic_cnn_args}
# generate a name for the experiment
expe_name = '_'.join([f"{key}={val}" for key, val in hparams.items()])
print("Experimenting with : \n \t"+expe_name)
# path where to save the model
savepath = Path('models/checkpt.pt')
# Tensorboard summary writer
if gen_args['no_tensorboard']:
    writer = None
else:
    writer = SummaryWriter("runs/runs"+"_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+expe_name)
# start the experiment
checkpoint = CheckpointState(net, optimizer, savepath=savepath)
fit(checkpoint, criterion, embeddings, train_loader, test_loader, gen_args['epochs'], writer=writer)

if not gen_args['no_tensorboard']:
    writer.close()