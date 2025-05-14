import os
import pickle
import src.bag_utils as utils


if __name__ == '__main__':
    bags_folder = os.environ.get('BAGS_PATH')
    if not bags_folder:
        raise ValueError('BAGS_PATH is not set')
    if not os.path.isdir(bags_folder):
        raise ValueError(f'Unable to locate BAGS_PATH: {bags_folder}')
    
    gt_path = os.environ.get('GT_PATH')
    if gt_path and not os.path.isfile(gt_path):
        raise ValueError(f'Unable to locate GT_PATH: {gt_path}')

    tmp_file = os.path.join(bags_folder, 'processed_data.pkl')
    data = utils.process_bags(bags_folder=bags_folder, gt_path=gt_path)
    with open(tmp_file, 'wb') as f:
        pickle.dump(data, f)
