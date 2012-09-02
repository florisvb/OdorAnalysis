import sys, os
from optparse import OptionParser

import fly_plot_lib
fly_plot_lib.set_params.pdf()

import flydra_analysis_tools.flydra_analysis_dataset as fad
import prep_dataset

# plotting functions:
from plot_scripts import plot_heatmaps
from plot_scripts import plot_spagetti
from plot_scripts import plot_activity_histograms

def main(path, config, reprep=False):

    if reprep:
        culled_dataset = prep_dataset.main(path, config)
        dataset = culled_dataset
    else:
        culled_dataset_filename = os.path.join(path, config.culled_datasets_path, config.culled_dataset_name) 
        culled_dataset = fad.load(culled_dataset_filename)
        dataset = culled_dataset
    
    if 0:
        raw_dataset_name = os.path.join(path, config.raw_datasets_path, config.raw_dataset_name)
        raw_dataset = fad.load(raw_dataset_name)
        prep_dataset.prep_data(raw_dataset, path, config)
        dataset = raw_dataset    
    
    
    figure_path = os.path.join(path, config.figure_path)
    
    plot_heatmaps.main(config, dataset, save_figure_path=os.path.join(figure_path, 'heatmaps/') )
    plot_spagetti.main(config, dataset, save_figure_path=os.path.join(figure_path, 'spagetti/') )
    plot_activity_histograms.main(dataset, save_figure_path=os.path.join(figure_path, 'activity/') )
    
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--path", type="str", dest="path", default='',
                        help="path to empty data folder, where you have a configuration file")
    (options, args) = parser.parse_args()
    
    path = options.path    
    sys.path.append(path)
    import analysis_configuration
    config = analysis_configuration.Config()
    
    main(path, config)
