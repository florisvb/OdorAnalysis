import sys, os
from optparse import OptionParser

import flydra_analysis_tools as fat


def get_h5_filelist(path, kalmanized=False):
    cmd = 'ls ' + path
    ls = os.popen(cmd).read()
    all_filelist = ls.split('\n')
    try:
        all_filelist.remove('')
    except:
        pass
        
    filelist = []
    for i, filename in enumerate(all_filelist):
        if filename[-3:] != '.h5':
            pass
        else:
            if kalmanized:
                if filename[-13:] == 'kalmanized.h5':
                    filelist.append(path + filename)
            else:
                if filename[-13:] == 'kalmanized.h5':
                    pass
                else:
                    filelist.append(path + filename)
    
    print
    print 'files: '
    for filename in filelist:
        print filename
    
    return filelist    
    
    


def main(config):
    
    # some options
    kalman_smoothing = config.kalman_smoothing
    save_covariance = config.save_covariance
    kalmanized = config.kalmanized # use kalmanized files?
    
    # path stuff
    savename = config.path_to_raw_dataset
    h5_path = config.path_to_h5
    tmp_path = config.path_to_tmp_data
    
    if config.h5_files == 'all':
    
        try:
            raw_dataset = fad.load(savename)
            print 'loaded pre-existing raw dataset!'
            filelist = get_h5_filelist(h5_path, kalmanized=kalmanized)
            
            for filename in filelist:
                if filename not in raw_dataset.h5_files_loaded:
                    raw_dataset.load(filename)
            
        except:
            print 'could not find or load raw_dataset -- loading all h5'
            fat.flydra_analysis_dataset.load_all_h5s_in_directory(h5_path, print_filenames_only=False, kalmanized=kalmanized, savedataset=True, savename=savename, kalman_smoothing=kalman_smoothing, dynamic_model=None, fps=None, info=config.info, save_covariance=save_covariance, tmp_path=tmp_path)

    else:
        dataset = fat.flydra_analysis_dataset.Dataset()
        for h5_file in config.h5_files:
            if config.h5_path not in h5_file:
                h5_file = os.path.join(path, config.h5_path, h5_file)
            dataset.load_data(h5_file, kalman_smoothing=kalman_smoothing, dynamic_model=None, fps=None, info=config.info, save_covariance=save_covariance)

        dataset.save(savename)
        


if __name__ == '__main__':
    
    parser = OptionParser()
    parser.add_option("--path", type="str", dest="path", default='',
                        help="path to empty data folder, where you have a configuration file")
    (options, args) = parser.parse_args()
    
    path = options.path
    sys.path.append(path)
    import analysis_configuration
    config = analysis_configuration.Config(path)
    
    main(config)
    
