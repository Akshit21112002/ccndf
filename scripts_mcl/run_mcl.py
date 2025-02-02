import tqdm
import glob
#import natsort
from loc_ndf.utils import particle_filter
import click
import torch

import matplotlib.pyplot as plt
from loc_ndf.mcl2d import models
import numpy as np
torch.manual_seed(123)


def transform(scans, poses):
    return (poses @ scans.transpose(-2, -1)).transpose(-2, -1)


def to_plt(pcd, T_local, img):
    part = pcd.clone()
    part[:, 2] = 1
    part = transform(part, T_local).cpu() * torch.tensor([*img.shape, 1])
    return part


def plot_results(figure, pfe, scan, T_local, img, gt_position, ylim):
    # pass
    part = to_plt(pfe.particles[:, :3], T_local, img)
    sort = torch.argsort(pfe.particles[:, -1]).cpu()
    colors = pfe.particles[:, -1].cpu()[sort]
    max_c = colors.max()
    plt.scatter(
        part[sort, 0], part[sort, 1], c=colors, cmap="YlOrRd", s=2, label='Particles', norm=plt.Normalize(-max_c, max_c))
    plt.plot(gt_position[0], gt_position[1], "xk",
             markersize=7, label='GT Position')
    plt.imshow(img, origin="lower")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(0, ylim)
    scan_plt = to_plt(transform(scan, pfe.get_pose()), T_local, img)
    plt.plot(scan_plt[:, 0], scan_plt[:, 1], ".k", label="Scan")

    plt.title(f"Monte Carlo Localization in NDF")
    plt.legend(loc="upper right")
    plt.savefig('output_plot.png')  # Save the plot to a file
    plt.close()  # Close the figure to release resources



def get_background_image(model, num_voxels):
    nv = [num_voxels, num_voxels]
    xy = model.get_grid(nv)  # ,file=file)
    img, valid = model.compute_distance(xy, in_global_frame=False)
    img = img.squeeze()
    img[~valid] = img.max()
    img = img.cpu().numpy().T
    return img


@click.command()
# Add your options here
@click.option(
    "--checkpoint",
    "-c",
    type=str,
    help="path to checkpoint file (.ckpt)",
    required=True,
)
@click.option(
    "--input_folder",
    "-i",
    type=str,
    help="path to dataset",
    required=True,
)
@click.option(
    "--calibration",
    "-cal",
    type=str,
    help="path to calibration file",
    required=True,
)
@click.option("--num_voxels", "-v", type=int, default=1000, required=True)
@click.option(
    "--output_file",
    "-o",
    type=str,
    help="output trajectory file",
    required=True,
)
def main(checkpoint, input_folder, num_voxels, calibration, output_file):
    errors_5cm = []
    errors_10cm = []
    errors_20cm = []
    errors_30cm = []
    errors_40cm = []
    errors_50cm = []
    total_errors = []
    total_errors_list=[]
    converged_runs = 0
    cfg = torch.load(checkpoint)["hyper_parameters"]
    folder = input_folder
    start_index = 0
    plt.ion()
    figure, _ = plt.subplots(figsize=(10, 8))

    # Load data and model
    model = models.MCLNet.load_from_checkpoint(checkpoint, hparams=cfg).cuda()
    model.requires_grad_(False)
    print(f'memory {model.get_memory():.2} MB')
    T_local = model.T_local
    bb_box = torch.tensor(
        model.hparams["bounding_box"], device=T_local.device,)
    calibration = torch.tensor(
        np.loadtxt(calibration), dtype=torch.float32
    ).cuda()  # n,3
    base2laser = torch.eye(3, device="cuda")
    base2laser[:2, :2] = calibration[:2, :2]
    base2laser[:2, -1] = calibration[:2, -1]

    # Set up particle filter
    pfe = particle_filter.ParticleFilter(
        int(3e4), bounding_box=bb_box, output_file=output_file, extrinsic=base2laser
    )

    poses_base_link = torch.tensor(
        np.loadtxt(f"{folder}/poses_gt.txt"), dtype=torch.float32
    ).cuda()  # n,3
    poses = particle_filter.log(
        particle_filter.exp(poses_base_link) @ base2laser)
    odom = torch.tensor(
        np.loadtxt(f"{folder}/odometry.txt"), dtype=torch.float32
    ).cuda()  # n,3
    odometry_se2 = particle_filter.exp(odom)
    odometry = torch.stack([torch.linalg.inv(odometry_se2[i - 1]) @ odometry_se2[i]
                            for i in range(start_index, len(odometry_se2))], dim=0,)

    pfe.resample_outside_points(model, bb_box)

    img = get_background_image(model, num_voxels)
    gt = to_plt(poses, T_local, img)

    scan_files = sorted(glob.glob(f"{folder}/scans/*.npy"))
    odom = particle_filter.log(odometry)

    for i, scan_file in enumerate(tqdm.tqdm(scan_files[start_index:])):
        index = start_index + i
        scan = torch.tensor(np.load(scan_file),
                            device="cuda", dtype=torch.float32).T
        pfe.apply_motion_model(odometry=odom[i])
        # resample point outside the area
        pfe.resample_outside_points(model, bb_box)

        if pfe.has_moved_enough():

            # Transform points
            scans_particle = pfe.scan_in_particles_frame(scan)
            # Get distance from NDF
            dist, valid = model.compute_distance(
                scans_particle, batch_size=int(1e3))
            dist = dist.squeeze()
            # Set distance from outside points to max
            dist[~valid] = dist.max()
            # Observation model
            pfe.apply_observation_model(distances=dist.abs())
            # Plot results
            plot_results(figure, pfe, scan, T_local, img,
                         gt[index], ylim=num_voxels//2)

            if pfe.is_in_pose_tracking():
                error = particle_filter.log(pfe.mean()[None, :]) - poses[index]
                error_norm = error[0, :2].norm().item()

                total_errors.append(error_norm)
                # print(f"Error: {error_norm:.3f}m")
                print(total_errors)
                # Check thresholds and accumulate errors
                if error_norm <= 5:  # 5 cm threshold
                    errors_5cm.append(error_norm)
                if error_norm <= 10:   # 10 cm threshold
                    errors_10cm.append(error_norm)
                if error_norm <= 20:   # 20 cm threshold
                    errors_20cm.append(error_norm)
                if error_norm <= 30:   # 30 cm threshold
                    errors_30cm.append(error_norm)
                if error_norm <= 40:   # 40 cm threshold
                    errors_40cm.append(error_norm)
                if error_norm <= 50:   # 50 cm threshold
                    errors_50cm.append(error_norm)
                else:
                    total_errors_list.append(error_norm)
                
                
                
        pfe.write_pose()
    calculate_and_print_rmse(errors_5cm, "5 cm")
    calculate_and_print_rmse(errors_10cm, "10 cm")
    calculate_and_print_rmse(errors_20cm, "20 cm")
    calculate_and_print_rmse(errors_20cm, "30 cm")
    calculate_and_print_rmse(errors_20cm, "40 cm")
    calculate_and_print_rmse(errors_20cm, "50 cm")
    calculate_and_print_rmse(total_errors, "total")
    calculate_and_print_mae(total_errors, "total")
    # print("Errors: ",error)
    print("Error List: ",total_errors_list)

def calculate_and_print_mae(errors, threshold):
    if len(errors) > 0:
        mae = np.mean((errors))
        # print(f"mae at {threshold} threshold: {rmse:.3f}m")
        print(f"Mean Absolute Error within {threshold}: {mae:.3f}cm")
    else:
        print(f"No errors within {threshold} threshold mae.")
def calculate_and_print_rmse(errors, threshold):
    if len(errors) > 0:
        rmse = np.sqrt(np.mean(np.square(errors)))
        # print(f"RMSE at {threshold} threshold: {rmse:.3f}m")
        print(f"Root Mean Square Error within {threshold}: {rmse:.3f}cm")
    else:
        print(f"No errors within {threshold} threshold.")

if __name__ == "__main__":
    with torch.no_grad():
        main()
