"""
utility functions for plant dataset
"""
dai_dict = dict({
    'Z': {
        'dai_offset': 9,
        1: {
            1: -1, 2: -1, 3: -1,
            4: 9, 5: 9, 6: 9, 7: 9, 8: 9,
            9: 14, 10: 14, 11: 14, 12: 14, 13: 14,
            14: 19, 15: 19, 16: 19, 17: 19, 18: 19,
        },
    },
})


# mapping the labels of incomplete dataset
dai_incom_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 6: 5, 7: 6, 8: 7, 9: 8, 12: 9, 13: 10, 14: 11}
dai_incom_dict_inv = {value: key for key, value in dai_incom_dict.items()}


for dai in range(2, 6):
    dai_dict['Z'][dai] = dai_dict['Z'][1].copy()
    for i in dai_dict['Z'][dai].keys():
        if type(i) == int and i >= 4:
            dai_dict['Z'][dai][i] = dai_dict['Z'][dai][i] + dai - 1

def get_dai_label(sample_id):
	"""
	get day after incubation given a string of the sample ID.
	sample_id e.g. '1,Z12,...'
	"""
	sample_id = sample_id.split(",")
	# sample_id e.g. '1,Z12,...'
	day = sample_id[0]
	plant_type = sample_id[1][0]
	sample_num = sample_id[1][1:]
	label = dai_dict[plant_type][int(day)][int(sample_num)]
	if label == -1:
		return 0
	else:
		return label + 1 - dai_dict[plant_type]['dai_offset']
