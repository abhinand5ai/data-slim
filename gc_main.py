
from main import create_argparser
from utils import logger, utils
from models import hierachical_res_2d
from torchinfo import summary
import torch
import time
import errno
import os
import data_io
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

    for i, (x, mask) in enumerate(ds):
        print(x, mask)


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

if __name__ == '__main__':
    print("parsng args")
    args = create_argparser()
    main(args)
