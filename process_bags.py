import os
import pickle
import src.bag_utils as utils

BAGS_FOLDER = '/data/deployment_nov2023'
TMP_FILE = os.path.join(BAGS_FOLDER, 'processed_data.pkl')


if __name__ == '__main__':
    gt_path = os.path.join(BAGS_FOLDER, 'GT_Nov2023.csv')
    data = utils.process_bags(bags_folder=BAGS_FOLDER, gt_path=gt_path)
    with open(TMP_FILE, 'wb') as f:
        pickle.dump(data, f)
