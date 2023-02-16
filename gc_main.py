
from main import create_argparser
from utils import logger, utils
from models import hierachical_res_2d
from torchinfo import summary
import torch
from torch import nn
import time
import errno
import os
import data_io
import compression
import shutil
from pathlib import Path
import glob
import netcdf_utils
import poptorch
# def create_argparser():
#     """Parses command line arguments"""
#     return {}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if DEVICE.type != "cpu":
    NUM_GPUS = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])
else:
    NUM_GPUS = 0
os.environ["DEVICE"] = str(DEVICE.type)
os.environ["NUM_GPUS"] = str(NUM_GPUS)

def main(args):
    logger.configure(dir="./tmp_logs")
    utils.configure_args(args)
    utils.log_args_and_device_info(args)
    model = get_model(args)
    # if args.verbose:
    #     log_model_summary(model)

    if args.command in ["compress"]:
        compress(args, model)
        
        
        
def compress(args, model):
    load_model_checkpoint(model, args.model_path)
    model.eval()
    model.name = args.model_path.split("/")[-3]
    dataio = data_io.Dataio(args.batch_size, args.patch_size, args.data_shape)
    dataio.batch_size = int(dataio.get_num_batch_per_time_slice())
    ds = dataio.get_compression_data_loader(args.input_path, args.ds_name)
    
    if args.verbose: data_io.log_training_parameters()
    
    logger.log("Compressing...")
    start_time = time.perf_counter()
    compress_loop(args, model, ds, dataio)
    logger.info(f"Compression completed!")
    logger.log(
        f"Total compression time: {time.perf_counter() - start_time:0.4f} seconds"
    )


def compress_loop(args, model, ds, dataio):
    output_path = create_output_location(args.output_path)

    logger.log("Making the metadata...")


    for i, (x, mask) in enumerate(ds):  
        output_filename = args.input_path.split("/")[-1].rpartition(".")[0] + f"_{i}"
        output_file = os.path.join(output_path, output_filename)
        if i == 0:
            logger.log("Making metadata...")
            meta_time = time.perf_counter()
            # Mask
            mask_path = os.path.join(output_path, "mask")
            compression.save_compressed(mask_path, [mask])
            # Stats
            stat_folder = Path(args.input_path).parent.absolute()
            stat_path = glob.glob(os.path.join(stat_folder, "*.csv"))[0]
            stat_path = Path(stat_path)
            stat_filename = "stats.csv"
            shutil.copy(stat_path, os.path.join(output_path, stat_filename))
            # Metadata
            metadata_output_file = output_filename[:-1] + "metadata.nc"
            metadata_output_file = os.path.join(output_path, metadata_output_file)
            if args.verbose:
                logger.log(f"Saving mask to {mask_path}")
                logger.log(f"Saving statistics to {stat_path}")
                logger.log(f"Saving metadata to {metadata_output_file}")
            netcdf_utils.create_dataset_with_only_metadata(
                args.input_path, metadata_output_file, args.ds_name, args.verbose
            )
            logger.log(
                f"Metadata completed in {time.perf_counter() - meta_time:0.4f} seconds"
            )
        if args.verbose:
            logger.log(f"\nCompressing {args.input_path}_{i}")
            tensors, x_hat = compression.compress(model, x, mask, args.verbose)

            # Save images of original data and reconstructed data for comparison.
            x = x * mask
            x = torch.permute(x, (0, 2, 3, 1)).detach().cpu().numpy()
            x = dataio.revert_partition(x)

            x_hat = x_hat * mask
            x_hat = torch.permute(x_hat, (0, 2, 3, 1)).detach().cpu().numpy()
            x_hat = dataio.revert_partition(x_hat)
            utils.save_reconstruction(x[0], x_hat[0], output_filename, output_file)
        else:
            # tensors = compression.compress(model, x, mask, args.verbose)
            tensors = gc_compress(model, x, mask)


        compression.save_compressed(output_file, tensors)
    
def gc_compress(model, x, mask=None):
    batch_size = int(os.environ.get("BATCH_SIZE", "8"))
    tensors = gc_compress_step(model, x, batch_size=batch_size)
    return tensors
    


def gc_compress_step(model, x, batch_size=4):
    opts = poptorch.Options()
    opts.deviceIterations(10)
    x = poptorch.DataLoader(options=opts,
                                        dataset=x,
                                        batch_size=10,
                                        shuffle=True,
                                        drop_last=True)
    gc_model = poptorch.inferenceModel(CompressionWrapper(model))
    z_tensor, y_tensor = [], []
    # model_device = next(model.parameters()).device
    for i, da in enumerate(x):
        compressed = model.compress(da.type(torch.float))
        y_tensor.append(compressed[0].detach().cpu())
        z_tensor.append(compressed[1].detach().cpu())
    y_tensor = torch.cat(y_tensor, axis=0).cpu()
    z_tensor = torch.cat(z_tensor, axis=0).cpu()
    tensors = (y_tensor, z_tensor, *compressed[2:])
    return tensors

    


def create_output_location(output_path):
    if not args.output_path:
        output_path = os.path.join("./outputs", "".join(args.model_path.split("/")[-3]))
    else:
        output_path = args.output_path
    utils.mkdir_if_not_exist(output_path)
    return output_path

def load_model_checkpoint(model, checkpoint_path):
    model, is_weight_loaded = utils.load_model_with_checkpoint(
            model, args.model_path, args.verbose
        )
    if not is_weight_loaded:
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), args.model_path
        )



def get_model(args):
    start_time = time.perf_counter()
    model = None
    if "hierachical" in args.model_type.lower():
        model = hierachical_res_2d.VQCPVAE(
            **utils.args_to_dict(args, utils.model_defaults().keys())
        )
    model = model.to(torch.device(DEVICE))
    stats = get_stats(args)
    mean = stats["mean"]
    std = stats["std"]
    model.set_standardizer_layer(mean, std**2, 1e-6)
    logger.log(f"Model initialization time: {time.perf_counter() - start_time:0.4f} seconds\n")
    return model

def get_stats(args):
    stats = None
    stats_dir = args.data_dir if args.data_dir else args.input_path
    if stats_dir:
        logger.info("Loading stats from {}".format(stats_dir))
        try:
            stats = utils.get_netcdf_data_stats(stats_dir)
        except Exception as e:
            logger.error("Failed to load stats: {}".format(e))
    if not stats:
        raise ValueError("No Statistics file available at {}".format(stats_dir))
    return stats

def log_model_summary(model):
    logger.log(
            summary(
                model,
                model.input_shape,
                depth=5,
                col_names=(
                    "input_size",
                    "output_size",
                    "num_params",
                ),
                verbose=args.verbose,
            )
        )
    
class CompressionWrapper(nn.Module):
    def __init__(self, compressionModel) -> None:
        super().__init__()
        self.model = compressionModel

    def forward(self, x):
        compressed = self.model.compress(x)
        return compressed

if __name__ == '__main__':
    print("parsng args")
    args = create_argparser()
    main(args)
