



def calc_casts_and_surges(trajec, threshold_odor=10):

    if np.max(trajec.odor < threshold_odor):
        return

    frames_where_odor = np.where(trajec.odor > threshold_odor)[0]
