from __future__ import with_statement
import motmot.ufmf.ufmf as ufmf_mod
import motmot.FlyMovieFormat.FlyMovieFormat as fmf_mod
import sys, os, tempfile, re, contextlib, warnings
from optparse import OptionParser
import flydra.a2.auto_discover_ufmfs as auto_discover_ufmfs
import numpy as np
import tables
import flydra.a2.utils as utils
import flydra.analysis.result_utils as result_utils
import scipy.misc
import subprocess
import motmot.imops.imops as imops

from flydra.a2.tables_tools import openFileSafe


def get_flydra_frame_number_for_ufmf_frame_number(ufmf_frame):
    
    camn = 32
    h5_filename = '/home/caveman/DATA/20120924_HCS_odor_horizon/data/h5_files/DATA20121002_184808.h5'
    ufmf_fnames = ['/home/caveman/DATA/20120924_HCS_odor_horizon/data/ufmfs/small_20121002_184626_Basler_21111538.ufmf']
    white_background=False
    max_n_frames = None
    start = None
    stop = None
    rgb8_if_color=False
    movie_cam_ids=['Basler_21111538']
    camn2cam_id = None
    
    with openFileSafe( h5_filename, mode='r' ) as h5:
        if camn2cam_id is None:
            camn2cam_id, cam_id2camns = result_utils.get_caminfo_dicts(h5)
        parsed = result_utils.read_textlog_header(h5)
        flydra_version = parsed.get('flydra_version',None)
        if flydra_version is not None and flydra_version >= '0.4.45':
            # camnode.py saved timestamps into .ufmf file given by
            # time.time() (camn_receive_timestamp). Compare with
            # mainbrain's data2d_distorted column
            # 'cam_received_timestamp'.
            old_camera_timestamp_source = False
            timestamp_name = 'cam_received_timestamp'
        else:
            # camnode.py saved timestamps into .ufmf file given by
            # camera driver. Compare with mainbrain's data2d_distorted
            # column 'timestamp'.
            old_camera_timestamp_source = True
            timestamp_name = 'timestamp'

        h5_data = h5.root.data2d_distorted[:]
        
        
    h5_indices_for_camn = np.where(h5_data['camn']==camn)[0]
    flydra_frames_for_camn = h5_data['frame'][h5_indices_for_camn]
    unique_flydra_frames_for_camn, unique_h5_indices_for_camn_idx = np.unique(flydra_frames_for_camn, return_index=True)
    unique_h5_indices_for_camn = h5_indices_for_camn[unique_h5_indices_for_camn_idx]
    unique_flydra_timestamps = h5_data[timestamp_name][unique_h5_indices_for_camn]
    
    maxn = 50
    n = 0
    for unique_flydra_index, unique_flydra_timestamp in enumerate(unique_flydra_timestamps):
        ufmf_frame_number = np.argmin( np.abs(movie.get_all_timestamps() - unique_flydra_timestamp ) )
        flydra_frame_number = unique_flydra_frames_for_camn[unique_flydra_index]
        print ufmf_frame_number - flydra_frame_number - h5_data['frame'][0], np.abs(movie.get_all_timestamps()[ufmf_frame_number] - unique_flydra_timestamp)
        n += 1
        if n > maxn:
            break
            

for frame in unique_flydra_frames_for_camn:
    ufmf_frame = frame-1618
    try:
        print frame_to_key[frame], frame, frame - dataset.trajecs[frame_to_key[frame][0]].first_frame, dataset.trajecs[frame_to_key[frame][0]].timestamp_epoch + dataset.trajecs[frame_to_key[frame][0]].time_fly[frame-dataset.trajecs[frame_to_key[frame][0]].first_frame], movie.get_frame(ufmf_frame)[1], movie.get_frame(ufmf_frame)[1] -  (dataset.trajecs[frame_to_key[frame][0]].timestamp_epoch + dataset.trajecs[frame_to_key[frame][0]].time_fly[frame-dataset.trajecs[frame_to_key[frame][0]].first_frame])
    except:
        pass
        
        
for body_key, body_trajec in body_trajectories.items():
    ufmf_frame = body_trajec['ufmf_frames'][0]
    try:
        print frame_to_key[ufmf_frame]
    except:
        pass

def iterate_frames(h5_filename,
                   ufmf_fnames, # or fmfs
                   white_background=False,
                   max_n_frames = None,
                   start = None,
                   stop = None,
                   rgb8_if_color=False,
                   movie_cam_ids=None,
                   camn2cam_id = None,
                   ):
    """yield frame-by-frame data"""

    h5_filename = '/home/caveman/DATA/20120924_HCS_odor_horizon/data/h5_files/DATA20121002_184808.h5'
    ufmf_fnames = ['/home/caveman/DATA/20120924_HCS_odor_horizon/data/ufmfs/small_20121002_184626_Basler_21111538.ufmf']
    white_background=False
    max_n_frames = None
    start = None
    stop = None
    rgb8_if_color=False
    movie_cam_ids=['Basler_21111538']
    camn2cam_id = None

    # First pass over .ufmf files: get intersection of timestamps
    first_ufmf_ts = -np.inf
    last_ufmf_ts = np.inf
    ufmfs = {}
    cam_ids = []
    for movie_idx,ufmf_fname in enumerate(ufmf_fnames):
        if movie_cam_ids is not None:
            cam_id = movie_cam_ids[movie_idx]
        else:
            cam_id = get_cam_id_from_ufmf_fname(ufmf_fname)
        cam_ids.append( cam_id )
        kwargs = {}
        if ufmf_fname.lower().endswith('.fmf'):
            ufmf = fmf_mod.FlyMovie(ufmf_fname)
        else:
            ufmf = ufmf_mod.FlyMovieEmulator(ufmf_fname,
                                             white_background=white_background,
                                             **kwargs)
        tss = ufmf.get_all_timestamps()
        ufmfs[ufmf_fname] = (ufmf, cam_id, tss)
        min_ts = np.min(tss)
        max_ts = np.max(tss)
        if min_ts > first_ufmf_ts:
            first_ufmf_ts = min_ts
        if max_ts < last_ufmf_ts:
            last_ufmf_ts = max_ts

    #assert first_ufmf_ts < last_ufmf_ts, ".ufmf files don't all overlap in time"

    #ufmf_fnames.sort()
    #cam_ids.sort()

    with openFileSafe( h5_filename, mode='r' ) as h5:
        if camn2cam_id is None:
            camn2cam_id, cam_id2camns = result_utils.get_caminfo_dicts(h5)
        parsed = result_utils.read_textlog_header(h5)
        flydra_version = parsed.get('flydra_version',None)
        if flydra_version is not None and flydra_version >= '0.4.45':
            # camnode.py saved timestamps into .ufmf file given by
            # time.time() (camn_receive_timestamp). Compare with
            # mainbrain's data2d_distorted column
            # 'cam_received_timestamp'.
            old_camera_timestamp_source = False
            timestamp_name = 'cam_received_timestamp'
        else:
            # camnode.py saved timestamps into .ufmf file given by
            # camera driver. Compare with mainbrain's data2d_distorted
            # column 'timestamp'.
            old_camera_timestamp_source = True
            timestamp_name = 'timestamp'

        h5_data = h5.root.data2d_distorted[:]

    if 1:
        # narrow search to local region of .h5
        #cond = ((first_ufmf_ts <= h5_data[timestamp_name]) &
        #        (h5_data[timestamp_name] <= last_ufmf_ts))
        #narrow_h5_data = h5_data[cond]
        
        narrow_h5_data = h5_data

        narrow_camns = h5_data['camn']
        narrow_timestamps = h5_data[timestamp_name]
        # Find the camn for each .ufmf file
        cam_id2camn = {}
        for cam_id in cam_ids:
            cam_id_camn_already_found = False
            for ufmf_fname in ufmfs.keys():
                (ufmf, test_cam_id, tss) = ufmfs[ufmf_fname]
                if cam_id != test_cam_id:
                    continue
                assert not cam_id_camn_already_found
                cam_id_camn_already_found = True

                umin=np.min(tss)
                umax=np.max(tss)
                cond = (umin<=narrow_timestamps) & (narrow_timestamps<=umax)
                ucamns = narrow_camns[cond]
                ucamns = np.unique(ucamns)
                camns = []
                for camn in ucamns:
                    if camn2cam_id[camn]==cam_id:
                        camns.append(camn)

                assert len(camns)<2, "can't handle multiple camns per cam_id"
                if len(camns):
                    cam_id2camn[cam_id] = camns[0]

        ff = utils.FastFinder(narrow_h5_data['frame'])
        unique_frames = list(np.unique(narrow_h5_data['frame']))
        unique_frames.sort()
        unique_frames = np.array( unique_frames )
        if start is not None:
            unique_frames = unique_frames[ unique_frames >= start ]
        if stop is not None:
            unique_frames = unique_frames[ unique_frames <= stop ]

        if max_n_frames is not None:
            unique_frames = unique_frames[:max_n_frames]
        for frame_enum,frame in enumerate(unique_frames):
            narrow_idxs = ff.get_idxs_of_equal(frame)

            # trim data under consideration to just this frame
            this_h5_data = narrow_h5_data[narrow_idxs]
            this_camns = this_h5_data['camn']
            this_tss = this_h5_data[timestamp_name]

            # a couple more checks
            if np.any( this_tss < first_ufmf_ts):
                continue
            if np.any( this_tss >= last_ufmf_ts):
                break

            per_frame_dict = {}
            for ufmf_fname in ufmf_fnames:
                ufmf, cam_id, tss = ufmfs[ufmf_fname]
                if cam_id not in cam_id2camn:
                    continue
                camn = cam_id2camn[cam_id]
                this_camn_cond = this_camns == camn
                this_cam_h5_data = this_h5_data[this_camn_cond]
                this_camn_tss = this_cam_h5_data[timestamp_name]
                if not len(this_camn_tss):
                    # no h5 data for this cam_id at this frame
                    continue
                this_camn_ts=np.unique1d(this_camn_tss)
                assert len(this_camn_ts)==1
                this_camn_ts = this_camn_ts[0]

                # optimistic: get next frame. it's probably the one we want
                try:
                    image,image_ts,more = ufmf.get_next_frame(_return_more=True)
                except ufmf_mod.NoMoreFramesException:
                    image_ts = None
                if this_camn_ts != image_ts:
                    # It was not the frame we wanted. Find it.
                    ufmf_frame_idxs = np.nonzero(tss == this_camn_ts)[0]
                    if (len(ufmf_frame_idxs)==0 and
                        old_camera_timestamp_source):
                        warnings.warn(
                            'low-precision timestamp comparison in '
                            'use due to outdated .ufmf timestamp '
                            'saving.')
                        # 2.5 msec precision required
                        ufmf_frame_idxs = np.nonzero(
                            abs( tss - this_camn_ts ) < 0.0025)[0]
                    assert len(ufmf_frame_idxs)==1
                    ufmf_frame_no = ufmf_frame_idxs[0]
                    image,image_ts,more = ufmf.get_frame(ufmf_frame_no,
                                                         _return_more=True)
                    del ufmf_frame_no, ufmf_frame_idxs
                coding = ufmf.get_format()
                if imops.is_coding_color(coding):
                    if rgb8_if_color:
                        image = imops.to_rgb8(coding,image)
                    else:
                        warnings.warn('color image not converted to color')
                per_frame_dict[ufmf_fname] = {
                    'image':image,
                    'cam_id':cam_id,
                    'camn':camn,
                    'timestamp':this_cam_h5_data['timestamp'][0],
                    'cam_received_timestamp':
                    this_cam_h5_data['cam_received_timestamp'][0],
                    'ufmf_frame_timestamp':this_cam_h5_data[timestamp_name][0],
                    }
                per_frame_dict[ufmf_fname].update(more)
            per_frame_dict['tracker_data']=this_h5_data
            yield (per_frame_dict,frame)
