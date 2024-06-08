import sys 
sys.path.insert(0, '../report_util/extract_Nripples/liset_tk')
from liset_tk import liset_tk

path = '../Amigo2_1_hippo_2019-07-11_11-57-07_1150um/'
liset = liset_tk(path, shank=3, start=30000000, numSamples=45000000, downsample=4000)

liset.plot_event([0, 5595426], offset=5, show_ground_truth=True, filtered=[100, 250], title='LFPs window')

model_path = '../trainSNNripples/optimized_model/256_128/E50_Y50/network.net'
liset.load_model(model_path)
liset.predict()