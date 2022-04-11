import os
import Network
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", type=str, default="datasets", help="Path ke datasets")
parser.add_argument("--attention_heads", type=int, default="16", help="Attention Heads")
parser.add_argument("--intermediate_size", type=int, default="8192", help="Intermediate Size")
parser.add_argument("--num_hidden_layers", type=int, default="16", help="Transfomer block size")
parser.add_argument("--rm_branch", type=str, default="SD", help="RM Branch")
parser.add_argument("--latent_dim", type=int, default="1024", help="Latent Dimension")
parser.add_argument("--max_seq_length", type=int, default="128", help="Maximum sequence length")

params = parser.parse_args()

if __name__ == '__main__':

    dataset_path = params.datasets
    attention_heads = params.attention_heads
    intermediate_size = params.intermediate_size
    num_hidden_layers = params.num_hidden_layers
    rm_branch = params.rm_branch
    latent_dim = params.latent_dim
    max_seq_length = params.max_seq_length

    Network.train(  dataset_path=dataset_path,  # path to the dataset folder containing the "Videos" foldes and "FileList.csv" file
                    num_epochs=5,               # number of epoch to train
                    device=[0],                 # "cpu" or gpu ids, ex [0] or [0,1] or [2] etc
                    batch_size=2,               # batch size
                    seed=0,                     # random seed for reproducibility
                    run_test=False,             # run test loop after each epoch
                    lr = 1e-5,                  # learning rate
                    modelname="UVT_repeat_reg",         # name of the folder where weight files will be stored
                    latent_dim=latent_dim,            # embedding dimension
                    lr_step_period=3,           # number of epoch before dividing the learning rate by 10
                    ds_max_length = max_seq_length,        # maximum number of frame during training
                    ds_min_spacing = 10,        # minimum number of frame during training
                    DTmode = 'repeat',          # data preprocessing method: 'repeat' (mirroring) / 'full' (entire video) / 'sample' (single heartbeat with random amounf of additional frames)
                    SDmode = 'reg',             # SD branch network type: reg (regression) or cla (classification)
                    num_hidden_layers = num_hidden_layers,     # Number of Transformers
                    intermediate_size = intermediate_size,   # size of the main MLP inside of the Transformers
                    rm_branch = rm_branch,           # select branch to not train: None, 'SD', 'EF'
                    use_conv = False,           # use convolutions instead of MLP for the regressors - worse results
                    attention_heads = attention_heads,        # number of attention heads in each Transformer
                    num_data = [200, 40]
                    )
    
    # Parameters must match train-time parameters, or the weight files wont load
    Network.test(   dataset_path=dataset_path,  # Path to the dataset folder containing the "Videos" foldes and "FileList.csv" file
                    SDmode='reg',               # SD branch network type: reg (regression) or cla (classification)
                    use_full_videos=True,       # Use full video (no preprocessing other than intensity scaling)
                    latent_dim=latent_dim,            # embedding dimension
                    num_hidden_layers=num_hidden_layers,       # Number of Transformers
                    intermediate_size=intermediate_size,     # Size of the main MLP inside of the Transformers
                    model_path="./output/UVT_repeat_reg",# path of trained weight
                    device=[0],
                    num_data = [40]
                    )
