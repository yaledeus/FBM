import mdtraj as md
import pymol
from pymol import cmd
import imageio
import argparse
import os

BASE_DIR = os.path.abspath('.')


def config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--traj', type=str, required=True, help='trajectory PDB file path')
    parser.add_argument('--step', type=int, default=1, help='time span between frames')
    parser.add_argument('--nframe', type=int, default=100, help='#frames in the video')

    return parser.parse_args()


def create_vis_dir():
    vis_dir = os.path.join(BASE_DIR, 'vis')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    return vis_dir


def main(traj_pdb, step=1, nframe=100):
    vis_dir = create_vis_dir()
    tmp = os.path.join(vis_dir, 'tmp')
    os.makedirs(tmp, exist_ok=True)

    traj_name = os.path.split(traj_pdb)[-1][:-4]

    FRAME_PREFIX = "movie_frame_"

    pymol.finish_launching()

    traj = md.load(traj_pdb)
    frames = []

    index = range(0, len(traj), step)[:nframe]

    for i in index:
        frame = traj[i]
        filename = os.path.join(tmp, f"frame_{i:04d}.pdb")
        frames.append(filename)
        frame.save_pdb(filename)

    cmd.delete('all')

    # load frames in PyMOL
    for f in frames:
        cmd.load(f, 'trajectory')

    # set PyMOL parameters
    cmd.hide('all')
    cmd.show('sticks')
    # set color
    cmd.set_color('res_color', [0.87, 0.6, 0.6])
    cmd.color('res_color', 'all')

    # create animation
    cmd.mset(f"1 -{len(frames)}")
    cmd.frame(1)
    cmd.mplay()

    # output to png
    cmd.mpng(FRAME_PREFIX)

    pngs = [os.path.join(BASE_DIR, f"{FRAME_PREFIX}{i:04d}.png") for i in range(1, len(index) + 1)]
    imageio.mimsave(os.path.join(vis_dir, f"{traj_name}-vis.gif"), [imageio.imread(png) for png in pngs], duration=0.1)

    # remove pngs
    for png in pngs:
        os.remove(png)
    # remove tmp directory
    os.remove(tmp)


if __name__ == "__main__":
    args = config()
    main(args.traj, step=args.step, nframe=args.nframe)
