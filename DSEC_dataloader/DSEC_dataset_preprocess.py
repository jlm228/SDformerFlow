import os
import imageio
# imageio.plugins.freeimage.download()
import numpy as np

from event_representations import EventSlicer
from event_representations import rectify_events, cumulate_spikes_into_frames, events_to_frames,VoxelGrid
import h5py
import torch
from tqdm import tqdm

'''
Adpated from "Optical Flow estimation from Event Cameras and Spiking Neural Networks"
https://github.com/J-Cuadrado/OF_EV_SNN
'''

def generate_files(root: str, sequence: str, events_input: str, num_frames_per_ts: int = 1 ):
    # Ground truth. This was commented out upstream, which meant the script produced event
    # tensors but no gt_tensors/ or mask_tensors/ at all, and training could not start.
    flow_path = os.path.join(root, 'train_optical_flow', sequence, 'flow', 'forward')

    save_path_flow = os.path.join(root, 'saved_flow_data', 'gt_tensors')
    save_path_mask = os.path.join(root, 'saved_flow_data', 'mask_tensors')

    # _create_flow_maps writes into these without creating them.
    os.makedirs(save_path_flow, exist_ok=True)
    os.makedirs(save_path_mask, exist_ok=True)

    _create_flow_maps(sequence, flow_path, save_path_flow, save_path_mask)


    timestamps = np.loadtxt(os.path.join(root, 'train_optical_flow', sequence, 'flow', 'forward_timestamps.txt'), delimiter = ',', dtype='int64')
    # timestamps = np.loadtxt(os.path.join(root, 'test_forward_optical_flow_timestamps', sequence + ".csv"), delimiter = ',', dtype='int64')

    eventsL_path = os.path.join(root, 'train_events', sequence, 'events', 'left')
    # eventsL_path = os.path.join(root, 'test_events', sequence, 'events', 'left')

    #save frames
    if events_input == "cnt":
        save_path_events = os.path.join(root, 'saved_flow_data', 'event_tensors',  '{}frames'.format(str(num_frames_per_ts).zfill(2)), 'left')
        print(save_path_events)
        _load_events(sequence, num_frames_per_ts, eventsL_path, timestamps, save_path_events,events_input)
    #save voxels
    if events_input == "voxel":
        # '10bins' not '10bins_pol': the active conversion below is convert_CHW (polarity as +/-
        # in one channel), and DSEC_dataset_lite maps loader.polarity=True -> '{n}bins'.
        # '{n}bins_pol' is the convert_CHW_polarities layout, which is not what this writes.
        save_path_events = os.path.join(root, 'saved_flow_data', 'event_tensors',  '{}bins'.format(str(num_frames_per_ts).zfill(2)), 'left')
        _load_events(sequence, num_frames_per_ts, eventsL_path, timestamps, save_path_events,events_input)

    if events_input == "list":
        save_path_events = os.path.join(root, 'saved_flow_data', 'event_tensors',  '{}lists'.format(str(num_frames_per_ts).zfill(2)), 'left')
        _load_events(sequence, num_frames_per_ts, eventsL_path, timestamps, save_path_events,events_input)

def _create_flow_maps(sequence: str, flow_maps_path, save_path_flow, save_path_mask):

    flow_maps_list = os.listdir(flow_maps_path)
    flow_maps_list.sort()

    img_idx = 0

    for flow_map in flow_maps_list:

        img_idx += 1

        path_to_flowfile = os.path.join(flow_maps_path, flow_map)

        flow_16bit = imageio.imread(path_to_flowfile, format='PNG-FI')

        flow_x = (flow_16bit[:,:,0].astype(float) - 2**15) / 128.
        flow_y = (flow_16bit[:,:,1].astype(float) - 2**15) / 128.
        valid_pixels = flow_16bit[:,:,2].astype(bool)


        flow_x = np.expand_dims(flow_x, axis=0)  # shape (H, W) --> (1, H, W)
        flow_y = np.expand_dims(flow_y, axis=0)


        flow_map = np.concatenate((flow_x, flow_y), axis = 0).astype(np.float32)

        filename = '{}_{}.npy'.format(sequence, str(img_idx).zfill(4))

        np.save(os.path.join(save_path_flow, filename), flow_map)
        np.save(os.path.join(save_path_mask, filename), valid_pixels)

def _load_events(sequence, num_frames_per_ts, events_path, timestamps, save_path_events,events_input):

    # load data
    datafile_path = os.path.join(events_path, "events.h5")
    datafile = h5py.File(datafile_path, 'r')
    event_slicer = EventSlicer(datafile)

    N_chunks = timestamps.shape[0]  # N_chunks = N_grountruths

    fileidx = 0

    # The rectification map is fixed per sequence. Upstream reopened the file and re-read the
    # whole array inside the per-chunk loop (thousands of redundant reads, and a handle leaked
    # on every iteration since only the last was ever closed). Read it once.
    rectmap_path = os.path.join(events_path, "rectify_map.h5")
    with h5py.File(rectmap_path, 'r') as rectmap_file:
        rectmap = rectmap_file['rectify_map'][()]


    if (events_input == "cnt"):
        for numchunk in tqdm(range(N_chunks)):

            fileidx += 1

            t_beg, t_end = timestamps[numchunk]
            dt = (t_end - t_beg) / num_frames_per_ts

            chunk = []

            for numframe in range(num_frames_per_ts):

                t_start = t_beg + numframe * dt
                t_end = t_beg + (numframe + 1) * dt

                # load events within time window
                event_data = event_slicer.get_events(t_start, t_end)

                p = event_data['p']
                t = event_data['t']
                x = event_data['x']
                y = event_data['y']

                # rectify events (rectmap hoisted above)
                xy_rect = rectify_events(x, y, rectmap)
                x_rect = xy_rect[:, 0]
                y_rect = xy_rect[:, 1]
                mask = (x_rect >= 0) & (x_rect < 640) & (y_rect >= 0) & (y_rect < 480)
                # cumulate events
                frame = cumulate_spikes_into_frames(x_rect[mask], y_rect[mask], p[mask])

                chunk.append(frame)

            # format into chunks
            chunk = np.array(chunk).astype(np.float32)
    else:
        for numchunk in tqdm(range(N_chunks)):

            fileidx += 1

            t_beg, t_end = timestamps[numchunk]

            dt = (t_end - t_beg) / num_frames_per_ts
            # chunk = []
            event_data = event_slicer.get_events(t_beg, t_end)
            p = event_data['p']
            t = event_data['t']
            x = event_data['x']
            y = event_data['y']
            # rectify events (rectmap hoisted above)
            xy_rect = rectify_events(x, y, rectmap)
            x_rect = xy_rect[:, 0]
            y_rect = xy_rect[:, 1]

            if (events_input == "cnt"):
                t = ((t - t_beg) / (t_end - t_beg))  # normalized t

                # t = (t - t[0]).astype('float32')
                # t = (t / t[-1])

                # frame = events_to_frames(p, x_rect,  y_rect, t, num_frames_per_ts)
                mask = (x_rect >= 0) & (x_rect < 640) & (y_rect >= 0) & (y_rect < 480)

                frame = cumulate_spikes_into_frames(x_rect[mask], y_rect[mask], p[mask])

                chunk.append(frame)
                # chunk=frame

            # generate list
            if (events_input == "list"):
                t = ((t - t_beg) / (t_end - t_beg))

                # t = (t - t[0]).astype('float32')
                # t = (t / t[-1])

                events_list = {'p': [], 't': [], 'x': [], 'y': []}
                events_list['p'] = p
                events_list['t'] = t
                events_list['x'] = x_rect
                events_list['y'] = y_rect

                chunk = events_list
                # format into chunks
                # chunk = np.array(chunk).astype(np.typeDict)

            # generate voxel
            if (events_input == "voxel"):
                t = (t - t[0]).astype('float32')
                t = (t / t[-1])

                voxel_grid = VoxelGrid((num_frames_per_ts, 480, 640))

                pol = p.astype('float32')
                event_data_torch = {
                    'p': torch.from_numpy(pol),
                    't': torch.from_numpy(t),
                    'x': torch.from_numpy(x_rect),
                    'y': torch.from_numpy(y_rect),
                }

                chunk = voxel_grid.convert_CHW(event_data_torch)
                # chunk = voxel_grid.convert_CHW_polarities(event_data_torch)
                # format into chunks
                # chunk = np.array(chunk).astype(np.typeDict)

            filename = '{}_{}.npy'.format(sequence, str(fileidx).zfill(4))
            # filename = '{}.npy'.format( str(index).zfill(6))
            save_path_dir = os.path.join(save_path_events, sequence)
            if not os.path.exists(save_path_dir):
                os.makedirs(save_path_dir)

            np.save(os.path.join(save_path_dir, filename), chunk)
            # np.save(os.path.join(save_path_dir, filename), chunk)
    # close hdf5 files (the rectify map was already closed by its context manager)
    datafile.close()



if __name__=='__main__':
    flow_sequences = [
                        'zurich_city_09_a',
                      'zurich_city_07_a',
                      'zurich_city_02_c',
                      'zurich_city_11_b',
                      'thun_00_a',
                      'zurich_city_02_d',
                      'zurich_city_11_c',
                      'zurich_city_03_a',
                      'zurich_city_10_a',
                      'zurich_city_05_b',
                      'zurich_city_08_a',
                      'zurich_city_01_a',
                      'zurich_city_10_b',
                      'zurich_city_02_e',
                      'zurich_city_05_a',
                      'zurich_city_06_a',
                      'zurich_city_11_a',
                      'zurich_city_02_a']
    test_sequences = ['interlaken_00_b',
                      'interlaken_01_a',
                      'thun_01_a',
                      'thun_01_b',
                      'zurich_city_12_a',
                      'zurich_city_14_c',
                      'zurich_city_15_a'
                    ]

    # Sequences may be named on the command line so a Slurm array can run one task each;
    # with no arguments the full 18-sequence list is processed serially, as before.
    #     python DSEC_dataset_preprocess.py                  # all 18
    #     python DSEC_dataset_preprocess.py thun_00_a        # one
    import sys

    selected = sys.argv[1:] or flow_sequences

    unknown = [s for s in selected if s not in flow_sequences]
    if unknown:
        raise SystemExit(
            'Unknown sequence(s): {}\nExpected one of:\n  {}'.format(
                ', '.join(unknown), '\n  '.join(flow_sequences)))

    for sequence in selected:
        print('=== {} ==='.format(sequence))
        generate_files(root = '../data/Datasets/DSEC', sequence = sequence, events_input = 'voxel', num_frames_per_ts = 10)
