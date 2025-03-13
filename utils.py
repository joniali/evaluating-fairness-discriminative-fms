import os
import math
import clip
import torch
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import utils_mitigate as ut_mitigate
from fair_pca import apply_fair_PCA_to_dataset
from sklearn.linear_model import LogisticRegression
from fair_pca_multi_group import apply_fair_PCA_to_dataset_multiple_groups



def get_features_ff(dataset, model, device):
	'''
	Given the Fairface dataset object, clip model and device this function returns all the features. 
	'''
	all_features = []
	all_labels_age = []
	all_labels_gender = []
	all_labels_race = []
	
	with torch.no_grad():
		
		for images, labels_age, labels_gender, labels_race in tqdm(DataLoader(dataset, batch_size=100)):
			
			features = model.encode_image(images.to(device))
			all_features.append(features)
			all_labels_age.append(labels_age)
			all_labels_gender.append(labels_gender)
			all_labels_race.append(labels_race)
			
	return torch.cat(all_features), torch.cat(all_labels_age).cpu().numpy(),\
			torch.cat(all_labels_gender).cpu().numpy(), torch.cat(all_labels_race).cpu().numpy()

def get_features_flickr(dataset, model, device):
	all_features = []
	all_labels_captions = []
	all_labels_gender = []
	
	with torch.no_grad():
		for images,captions, labels_gender in tqdm(DataLoader(dataset, batch_size=100)): 
			features = model.encode_image(images.to(device))
			
			all_features.append(features)
			all_labels_captions.append(captions)
			all_labels_gender.append(labels_gender)
			
	
	return torch.cat(all_features),np.concatenate(all_labels_captions, axis = 1).T, \
			torch.cat(all_labels_gender).cpu().numpy()

def calc_mutual_info(all_features_train, all_labels_gender_train):
	mis = []
	for col in range(all_features_train.shape[1]):
		mi = ut_mitigate.mutual_information_2d(all_features_train[:,col].squeeze().cpu().numpy(), all_labels_gender_train)
		mis.append((mi, col))
	mis = sorted(mis, reverse=False)
	mis = np.array([l[1] for l in mis])
	return mis 

def infer_feat_ff(feat,all_features_train, all_labels, model, device ):

	if feat == 'gender': 
		all_gender_labels = all_labels[feat]
		attr_task = ["woman", "man"]
		attribute_classification_inputs = torch.cat([clip.tokenize(f"a photo of a {word}") for word in attr_task]).to(device)
		with torch.no_grad():
			text_features = model.encode_text(attribute_classification_inputs)
		text_features /= text_features.norm(dim=-1, keepdim=True)
		
		similarity = (100.0 * all_features_train @ text_features.T).softmax(dim=-1).cpu().numpy().astype(np.float64)
		inferred_gender_train = np.argmax(similarity,axis=1)

		
		print(f"Error of predicting {feat} train = {np.around(np.mean(np.not_equal(all_gender_labels,inferred_gender_train)),2)}")

	elif feat == 'race':
		all_gender_labels = all_labels[feat]
		attr_task = ["black", "east asian", 'indian', 'latino hispanic', 'middle eastern' , 'southeast asian', 'white']
		attribute_classification_inputs = torch.cat([clip.tokenize(f"a photo of a {word} person") for word in attr_task]).to(device)
		with torch.no_grad():
			text_features = model.encode_text(attribute_classification_inputs)
		text_features /= text_features.norm(dim=-1, keepdim=True)
		
		similarity = (100.0 * all_features_train @ text_features.T).softmax(dim=-1).cpu().numpy().astype(np.float64)
		inferred_gender_train = np.argmax(similarity,axis=1)

		
		print(f"Error of predicting race train = {np.around(np.mean(np.not_equal(all_gender_labels,inferred_gender_train)),2)}")

	return inferred_gender_train

def calculate_projections_ff(clip_model, preprocess, device):

	import sys
	sys.path.insert(0, '../FairFace')
	import fair_face_dataset as ff
	import sys
	sys.path.insert(0, '../')
	from fair_pca_multi_group import apply_fair_PCA_to_dataset_multiple_groups

	fair_face_train_dataset = ff.FairFaceDataset('../../fairface_label_train.csv', '../../fairface-img-margin025-trainval', transform = preprocess)

	
	all_features_train, all_labels_age_train, all_labels_gender_train, all_labels_race_train  = get_features_ff(fair_face_train_dataset, clip_model, device)

	all_labels_train = {'age': all_labels_age_train, 'gender': all_labels_gender_train, 'race': all_labels_race_train}

	projection_pca_gt = {}
	projection_pca_inf = {}
	MI_gt_dict = {}
	MI_inf_dict = {}
	
	all_features_train /= all_features_train.norm(dim=-1, keepdim=True)

	for attr in ['gender', 'race']:
		
		all_feat_labels = all_labels_train[attr]


		MI_GT = calc_mutual_info(all_features_train, all_feat_labels)
		
		MI_gt_dict[attr] = MI_GT

		if len(np.unique(all_feat_labels)) > 2: 
			print(" unique attr", len(np.unique(all_feat_labels)))

			pipe_train = apply_fair_PCA_to_dataset_multiple_groups((all_features_train.cpu().numpy().astype(np.float64), all_labels_train['gender'], all_feat_labels),
				all_features_train.shape[1]-len(np.unique(all_feat_labels)) + 1, 
				LogisticRegression,
				'selection_rate_parity', 0, 
				standardize=False, fit_classifier=False)
		else:
			pipe_train = apply_fair_PCA_to_dataset((all_features_train.cpu().numpy().astype(np.float64), all_feat_labels, all_feat_labels),
				all_features_train.shape[1]- 1, 
				LogisticRegression,
				'selection_rate_parity', 0, 
				standardize=False, fit_classifier=False)

		projection_pca_gt[attr] = pipe_train

		inferred_att = infer_feat_ff(attr, all_features_train, all_labels_train, clip_model, device)

		MI_INF = calc_mutual_info(all_features_train, inferred_att)
		MI_inf_dict[attr] = MI_INF

		if len(np.unique(inferred_att)) > 2: 
			pipe_train_inferred = apply_fair_PCA_to_dataset_multiple_groups((all_features_train.cpu().numpy().astype(np.float64), all_labels_train['gender'], inferred_att),
				all_features_train.shape[1]- len(np.unique(all_feat_labels)) + 1, 
				LogisticRegression,
				'selection_rate_parity', 0, 
				standardize=False, fit_classifier=False)
		else:
			pipe_train_inferred = apply_fair_PCA_to_dataset((all_features_train.cpu().numpy().astype(np.float64), inferred_att, inferred_att),
			all_features_train.shape[1]-1, 
			LogisticRegression,
			'selection_rate_parity', 0, 
		standardize=False, fit_classifier=False)

		projection_pca_inf[attr] = pipe_train_inferred

	return projection_pca_gt, projection_pca_inf, MI_gt_dict, MI_inf_dict, all_features_train, all_labels_train

def run_linear_probe_ff(train_feat, test_feat, all_labels_train, all_labels_test, method):

	from sklearn.linear_model import LogisticRegressionCV
	from sklearn.linear_model import LogisticRegression

	features = ['age', 'gender', 'race']
	df = pd.DataFrame(columns = features)
	accs = []
	for f in features:
		classifier = LogisticRegression(random_state=0, C=1, max_iter=10000, verbose=0)
		#classifier = LogisticRegressionCV(random_state=0, max_iter=10000, verbose=0, n_jobs = 45, solver = 'liblinear')

		classifier.fit(train_feat, all_labels_train[f])

		predictions_test = classifier.predict(test_feat)
		
		accs.append(classifier.score(test_feat, all_labels_test[f]))

	df_row = pd.DataFrame([accs], columns= features)
	df = df.append(df_row, ignore_index=True)
	df = df.round(2)
	df.to_csv(f"../results_csv/{method}_linprobe.csv")
	print(df)

def calc_similarity_diff(method_name, attr, word_list, all_labels, attribute_dict, similarities ):
	'''
	This method calculates the difference of average similarity scores for gender and race attributes of fairface dataset. 
	'''
	print(f'--- Evaluation of mean similarity scores w.r.t. {attr} on Val ---')
	
	temp = np.zeros((len(word_list),len(attribute_dict.keys())))
	for cc in range (len(word_list)):
		for ell in attribute_dict.keys():
			temp[cc, ell] = np.around(np.mean(similarities[cc, all_labels[attr]==ell]),2)

	columns= list(attribute_dict.values())
	print(attr, columns)
	# [FairFace_val.attribute_to_integer_dict_inverse[attr][ell] for ell in range(FairFace_val.attribute_count_dict[attr])]
	temp = pd.DataFrame(temp, columns=columns, index=word_list)
	if attr == 'gender':	  
		temp['Disparity'] = temp['Male'] - temp['Female']
	elif attr == 'race':
		temp['Disparity'] = temp.max(axis = 1) - temp.min(axis = 1)
	# print(f"../results_csv/{attr}_{method_name}_disparities.csv")
	temp.to_csv(f"../results_csv/{attr}_{method_name}_disparities.csv")
	print(temp)
	print('-------------------------------------------------------------------')

def calc_min_max_skew(frac_returned, k, fracs_desired):
	'''
	It calculates the skew metric Eq.(4) in the paper
	'''
	skews = []
	for  frac_ret, frac_des in zip(frac_returned,fracs_desired):
		skews.append(np.log(frac_ret / frac_des))
	# print(skews)
	return min(skews), max(skews), max(abs(np.asarray(skews)))

def run_skew(queries, protected_attribute, similarities, fname, topks, skip_attr = None, sorted_idx = None):
	'''
	It calculates the skew metric Eq.(4) in the paper
	'''
	
	if sorted_idx == None:
		query_dict = {}
		for  idx, k in enumerate(queries):

			query_dict[k] = similarities[idx]
			
		sorted_idx = {}
		for k in query_dict.keys():
			s = np.asarray(query_dict[k])
			ind_sorted = s.argsort()
			sorted_idx[k] = []
			for topk in topks:
				ind = ind_sorted[-topk:][::-1]
				sorted_idx[k].append(ind) 


	col_names = []
	col_names.append('Query')		
	for k in topks:
		col_names.append(f'abs_skew_top_{k}')
	# col_test = ['Query', 'abs_skew_top_10', 'abs_skew_top_50', 'abs_skew_top_100']
	n_unique = np.unique(protected_attribute)
	desired_fracs =[]
	for r in n_unique:
		idx_group = np.where(protected_attribute == r)
		
		desired_fracs.append(1/len(n_unique))
	# print("desired_frac", desired_fracs)
	df_skews = pd.DataFrame(columns = col_names)
	for k_idx, (k, v) in enumerate(sorted_idx.items()):
		
		row = [k]
		for topi, v_ in enumerate(v):
			
			frac_ret = []
			
			for r in n_unique:
				
				if skip_attr != None:
					if r == skip_attr:
						continue

				num_r = np.where(protected_attribute[v_] == r)[0].shape[0]
				
				if num_r == 0:
					frac_ret.append(1e-2)
				else:
					frac_ret.append(num_r/topks[topi])
			min_S, max_S, abs_S = calc_min_max_skew(frac_ret, topks[topi], desired_fracs)
			row.append(abs_S)
		df_row = pd.DataFrame([row], columns= col_names)
		df_skews = df_skews.append(df_row, ignore_index=True)
	df_skews = df_skews.round(2)
	df_skews.to_csv(f"../results_csv/{fname}_skew.csv")
	print(df_skews)
	return df_skews

def run_anova(queries, protected_attribute, similarities, fname, skip_att = None):
	query_dict = {}
	for  idx, k in enumerate(queries):
		query_dict[k] = similarities[idx]

	col_test = ['Query', 'stat', 'pval']
	df_tests = pd.DataFrame(columns = col_test)
	for k_idx, (k, v) in enumerate(query_dict.items()):	
		n_unique = np.unique(protected_attribute)
		groups = []
		for r in n_unique:
			if r == skip_att:
				continue
			idx_group = np.where(protected_attribute == r)
			groups.append(v[idx_group])
		test = stats.alexandergovern(*groups)
		df_tests = df_tests.append({'Query': k, 'stat':test.statistic, 'pval': test.pvalue }, ignore_index=True)
	df_tests.to_csv(f"../results_csv/{fname}_statistical_test.csv")
	print(df_tests)
	return df_tests

def calculate_retrieval_metric(group_selected, K, group_num, total_num, print_ = False):
	'''
	This function calculates the retrieval metric given in equation 2 in the paper
	'''
	max_val = 0 
	if len(group_selected) == 2:
		max_val = group_selected[1] / K - ((group_num[1] - group_selected[1])/(total_num - K)) - (group_selected[0] / K - ((group_num[0] - group_selected[0])/(total_num - K)))
		# print(group_selected,K, group_num, total_num, max_val )

		# in some cases due to numerical approximations we get a value greater than 1 or less than -1
		max_val = max(max_val, -1)
		max_val = min(max_val, 1)

	else:
		
		for g_sele, g_num in zip(group_selected, group_num):
			for g_sele_2, g_num_2 in zip(group_selected, group_num):
				temp = abs((g_sele / K - ((g_num - g_sele)/(total_num - K))) - (g_sele_2 / K - ((g_num_2 - g_sele_2)/(total_num - K))))
				if temp > max_val:
					max_val = temp
		# in some cases due to numerical approximations we get a value greater than 1 
		max_val = min(max_val, 1)
		
	return max_val

def run_retrieval_metric(queries, protected_attribute, similarities, fname, topks, skip_attr = None, sorted_idx = None, find_person = False, all_cat = None):

	if sorted_idx is None:
		#construct a query dict with values as keys
		query_dict = {}
		for  idx, k in enumerate(queries):

			query_dict[k] = similarities[idx]
			

		# sorted_idx contains the sorted index for each query of top k items given by topks
		sorted_idx = {}
		for k in query_dict.keys():
			s = np.asarray(query_dict[k])
			ind_sorted = s.argsort()
			sorted_idx[k] = []
			for topk in topks:
				ind = ind_sorted[-topk:][::-1]
				sorted_idx[k].append(ind) 
		
	col_names = []
	col_names.append('Query')		
	for k in topks:
		col_names.append(f'ddp_top_{k}')
		if find_person:
			col_names.append(f'person_top_{k}')

	
	n_unique = np.sort(np.unique(protected_attribute))
	group_nums = []
	for r in n_unique:
		if skip_attr != None:
					if r == skip_attr:
						continue
		group_nums.append(np.where(protected_attribute == r)[0].shape[0])
	
	
	df_ddp = pd.DataFrame(columns = col_names)
	for k_idx, (k, v) in enumerate(sorted_idx.items()):
		
		row = [k]
		for topi, v_ in enumerate(v):
			if find_person:
				assert(all_cat is not None)
				person = sum(all_cat[v_][:,1] == 1)
			
			num_ret = []
			for r in n_unique:
				if skip_attr != None:
					if r == skip_attr:
						continue
				num_r = np.where(protected_attribute[v_] == r)[0].shape[0]
				
				num_ret.append(num_r)
			# print(num_ret)
			ret_metric = calculate_retrieval_metric(num_ret, topks[topi], group_nums, len(protected_attribute))
			row.append(ret_metric)
			if find_person:
				row.append(person)

		df_row = pd.DataFrame([row], columns= col_names)
		df_ddp = df_ddp.append(df_row, ignore_index=True)
	df_ddp = df_ddp.round(2)
	df_ddp.to_csv(f"../results_csv/{fname}_ret_met.csv")
	print(df_ddp)
	return df_ddp

def construct_queries_mixed(queries_orig, similarities_split, topks, num_protected):
	import math
	import random
	def get_top_k(topk, similarities):
		ind_sorted = similarities.argsort()	
		return ind_sorted[-topk:][::-1]

	random.seed(100)
	sorted_idx = {}
	for i in range(0,len(queries_orig)*num_protected,num_protected):
		sorted_idx[queries_orig[int(i/num_protected)]] = []
		print("query" , queries_orig[int(i/num_protected)])
		for topk in topks:
			topk_pro = []
			# to take care of scenarios when topk is not exactly divisible by num_protected 
			indices_additional = random.sample(range(num_protected),topk % num_protected)
			print("additional indices", topk, indices_additional)
			for idx, k in enumerate(range(num_protected)):
				print(i, k)
				if idx in indices_additional:
					topk_pro.extend(list(get_top_k(int(topk/num_protected) + 1, similarities_split[i + k])))
				else:
					topk_pro.extend(list(get_top_k(int(topk/num_protected), similarities_split[i + k])))
			# print(topk, len(topk_pro))
			assert(len(topk_pro) == topk)
			sorted_idx[queries_orig[int(i/num_protected)]].append(topk_pro)
	return sorted_idx

def run_skew_mixed(queries_orig, similarities_gendered, protected_attribute, fname, topks, skip_attr = None):

	if skip_attr != None:
		sorted_idx = construct_queries_mixed(queries_orig,similarities_gendered,topks, len(np.unique(protected_attribute))- 1)
	else:
		sorted_idx = construct_queries_mixed(queries_orig,similarities_gendered,topks, len(np.unique(protected_attribute)))

	return run_skew(queries_orig, protected_attribute, None, fname, topks,skip_attr = skip_attr, sorted_idx = sorted_idx )

def run_retrieval_metric_mixed(queries,  similarities,protected_attribute, fname, topks, skip_attr = None, find_person= False,all_cat = None ):
	if skip_attr != None:

		sorted_idx = construct_queries_mixed(queries,similarities,topks, len(np.unique(protected_attribute)) - 1)
	else:
		sorted_idx = construct_queries_mixed(queries,similarities,topks, len(np.unique(protected_attribute)))
	

	return run_retrieval_metric(queries, protected_attribute, None, fname, topks, skip_attr = skip_attr, sorted_idx = sorted_idx, find_person = find_person, all_cat = all_cat)


def calculate_recall(similarities, fname):
	# assuming that there are 5 captions or queries per image and they are in order in similarities 
	
	topks = [1, 5, 10]
	sorted_k_idx = []
	recall_k = dict.fromkeys(topks)

	for idx, s in enumerate(similarities):
		s = np.asarray(s)
		ind_sorted = s.argsort()
		
		for topk in topks:
			if idx == 0:
				recall_k[topk] = []
			ind = ind_sorted[-topk:][::-1]
			if int(idx/5) in ind:
				recall_k[topk].append(1)
			else:
				recall_k[topk].append(0)
	col_test = ['mean_top_1', 'mean_top_5',  'mean_top_10']
	row = []
	for topk in topks:
		row.append(np.mean(recall_k[topk]))

	df = pd.DataFrame([row], columns= col_test)
	df = df.round(3)
	df.to_csv(f"../results_csv/{fname}_captions_recall.csv")
	print(df)
	return df


def infer_gender(all_features_train,  all_gender_labels, model, device):
	attr_task = ["woman", "man", "object"]
	attribute_classification_inputs = torch.cat([clip.tokenize(f"a photo of a {word}") for word in attr_task]).to(device)
	with torch.no_grad():
		text_features = model.encode_text(attribute_classification_inputs)
	text_features /= text_features.norm(dim=-1, keepdim=True)
	
	similarity = (100.0 * all_features_train @ text_features.T).softmax(dim=-1).cpu().numpy().astype(np.float64)
	inferred_gender_train = np.argmax(similarity,axis=1)

	
	print(f"Error of predicting gender train = {np.around(np.mean(np.not_equal(all_gender_labels,inferred_gender_train)),2)}")

	return inferred_gender_train

def calculate_projections_flickr(all_features_train, all_feat_labels, clip_model, device):	
	

	projection_pca_gt = {}
	projection_pca_inf = {}
	MI_gt_dict = {}
	MI_inf_dict = {}

	for attr in ['gender']:

		MI_GT = calc_mutual_info(all_features_train, all_feat_labels)
		
		idxs = np.where(all_feat_labels!= 2)[0]
		
		MI_gt_dict[attr] = MI_GT

		
		pipe_train = apply_fair_PCA_to_dataset((all_features_train[idxs].cpu().numpy().astype(np.float64), all_feat_labels[idxs], all_feat_labels[idxs]),
				all_features_train[idxs].shape[1]- 1, 
				LogisticRegression,
				'selection_rate_parity', 0, 
				standardize=False, fit_classifier=False)

		projection_pca_gt[attr] = pipe_train

		inferred_att = infer_gender(all_features_train, all_feat_labels, clip_model, device)

		MI_INF = calc_mutual_info(all_features_train, inferred_att)
		MI_inf_dict[attr] = MI_INF
		idxs = np.where(inferred_att!= 2)[0]

		pipe_train_inferred = apply_fair_PCA_to_dataset((all_features_train[idxs].cpu().numpy().astype(np.float64), inferred_att[idxs], inferred_att[idxs]),
			all_features_train[idxs].shape[1]-1, 
			LogisticRegression,
			'selection_rate_parity', 0, 
		standardize=False, fit_classifier=False)

		projection_pca_inf[attr] = pipe_train_inferred

	return projection_pca_gt, projection_pca_inf, MI_gt_dict, MI_inf_dict

def get_features_coco(dataset, model, device):
	all_features = []
	all_labels_categories = []
	all_labels_captions = []
	all_labels_gender = []

	with torch.no_grad():
	
		for images, cat, cap, gender in tqdm(DataLoader(dataset, batch_size=100)):

			
			features = model.encode_image(images.to(device)).to(device)#, dtype=torch.float32)

				
			all_features.append(features)
			all_labels_categories.append(cat)
			all_labels_captions.append(cap)
			all_labels_gender.append(gender)
			

	return torch.cat(all_features), torch.cat(all_labels_categories).cpu().numpy(),\
		   np.concatenate(all_labels_captions, axis = 1).T, torch.cat(all_labels_gender).cpu().numpy()

def calculate_projections_coco(clip_model, preprocess, device):
	
	import sys
	sys.path.insert(0, 'MSCOCO/')
	sys.path.insert(0, '../MSCOCO/')

	import customCOCO as cusCoco #import MyCocoDataset

	mycoco_train = cusCoco.MyCocoDataset('../../MSCOCO/train2014','../../MSCOCO/annotations/instances_train2014.json', '../../MSCOCO/annotations/captions_train2014.json', transform = preprocess )
	
	all_features, all_labels_cat, all_labels_captions, all_gender_labels,  = get_features_coco(mycoco_train, clip_model, device)

	all_features /= all_features.norm(dim=-1, keepdim=True)

	MI_GT = calc_mutual_info(all_features, all_gender_labels)
	# print("norm shape: ", all_features.norm(dim=-1, keepdim=True).shape)
	# later do inferred from clip model
	idxs = np.where(all_gender_labels!= 2)[0]
	# print("dim fpca: ", all_features.shape, all_features.shape[1]-1)
	pipe_train = apply_fair_PCA_to_dataset((all_features[idxs].cpu().numpy().astype(np.float64), all_gender_labels[idxs], all_gender_labels[idxs]),
		all_features.shape[1]-1, 
		LogisticRegression,
		'selection_rate_parity', 0, 
		standardize=False, fit_classifier=False)
	
	inferred_gender = infer_gender(all_features,all_gender_labels, clip_model, device)
	MI_INF = calc_mutual_info(all_features, inferred_gender)

	idxs = np.where(inferred_gender!= 2)[0]
	pipe_train_inferred = apply_fair_PCA_to_dataset((all_features[idxs].cpu().numpy().astype(np.float64), inferred_gender[idxs], inferred_gender[idxs]),
		all_features.shape[1]-1, 
		LogisticRegression,
		'selection_rate_parity', 0, 
		standardize=False, fit_classifier=False)

	return pipe_train, pipe_train_inferred, MI_GT, MI_INF

def get_features_miap(dataset, model, device):
    all_features = []
    all_gender = []
    all_age = []
    all_area = []
    all_num_people = []
    with torch.no_grad():
        for images, gen, age, area, num_people in tqdm(DataLoader(dataset, batch_size=100)):
            
            features = model.encode_image(images.to(device))
            all_features.append(features)
            all_gender.append(gen)
            all_age.append(age)
            all_area.append(area)
            all_num_people.append(num_people)
    return torch.cat(all_features), torch.cat(all_gender).cpu().numpy(), torch.cat(all_area).cpu().numpy(), torch.cat(all_age).cpu().numpy(), torch.cat(all_num_people).cpu().numpy()

def calculate_projections_miap(clip_model, preprocess, device):
	
	from miap_dataset import MIAPDataset
	
	from fair_pca_multi_group import apply_fair_PCA_to_dataset_multiple_groups

	dataset = MIAPDataset("../../miap", "train", transform = preprocess)

	all_features_train, all_labels_gender_train, _, age, _,= get_features_miap(dataset,clip_model,device)
	
	idx_ = np.where(np.logical_and(all_labels_gender_train != 2, age != 2))
	all_features_train =all_features_train[idx_]
	all_labels_gender_train = all_labels_gender_train[idx_]

	all_labels_train = {'gender': all_labels_gender_train}

	projection_pca_gt = {}
	projection_pca_inf = {}
	MI_gt_dict = {}
	MI_inf_dict = {}
	
	all_features_train /= all_features_train.norm(dim=-1, keepdim=True)

	for attr in ['gender']:
		
		all_feat_labels = all_labels_train[attr]


		MI_GT = calc_mutual_info(all_features_train, all_feat_labels)
		
		MI_gt_dict[attr] = MI_GT

		if len(np.unique(all_feat_labels)) > 2: 
			print(" unique attr", len(np.unique(all_feat_labels)))

			pipe_train = apply_fair_PCA_to_dataset_multiple_groups((all_features_train.cpu().numpy().astype(np.float64), all_labels_train['gender'], all_feat_labels),
				all_features_train.shape[1]-len(np.unique(all_feat_labels)) + 1, 
				LogisticRegression,
				'selection_rate_parity', 0, 
				standardize=False, fit_classifier=False)
		else:
			pipe_train = apply_fair_PCA_to_dataset((all_features_train.cpu().numpy().astype(np.float64), all_feat_labels, all_feat_labels),
				all_features_train.shape[1]- 1, 
				LogisticRegression,
				'selection_rate_parity', 0, 
				standardize=False, fit_classifier=False)

		projection_pca_gt[attr] = pipe_train

		inferred_att = infer_feat_ff(attr, all_features_train, all_labels_train, clip_model, device)

		MI_INF = calc_mutual_info(all_features_train, inferred_att)
		MI_inf_dict[attr] = MI_INF

		if len(np.unique(inferred_att)) > 2: 
			pipe_train_inferred = apply_fair_PCA_to_dataset_multiple_groups((all_features_train.cpu().numpy().astype(np.float64), all_labels_train['gender'], inferred_att),
				all_features_train.shape[1]- len(np.unique(all_feat_labels)) + 1, 
				LogisticRegression,
				'selection_rate_parity', 0, 
				standardize=False, fit_classifier=False)
		else:
			pipe_train_inferred = apply_fair_PCA_to_dataset((all_features_train.cpu().numpy().astype(np.float64), inferred_att, inferred_att),
			all_features_train.shape[1]-1, 
			LogisticRegression,
			'selection_rate_parity', 0, 
		standardize=False, fit_classifier=False)

		projection_pca_inf[attr] = pipe_train_inferred

	return projection_pca_gt, projection_pca_inf, MI_gt_dict, MI_inf_dict#, all_features_train, all_labels_train

def run_relevance_celeba(similarities, queries, all_labels, desired_cat,  fname):
	topks = [20, 50, 100]
	query_dict = {}
	for  idx, k in enumerate(queries):

		query_dict[k] = similarities[idx]
		
	sorted_idx = {}
	for k in query_dict.keys():
		s = np.asarray(query_dict[k])
		ind_sorted = s.argsort()
		sorted_idx[k] = []
		for topk in topks:
			ind = ind_sorted[-topk:][::-1]
			sorted_idx[k].append(ind)

	col_test = ['Query', 'precision_top_20', 'precision_top_50', 'precision_top_100']
	df_skews = pd.DataFrame(columns = col_test)
    
        
	for k_idx, (k, v) in enumerate(sorted_idx.items()):
		
		row = [k]
		print_ = False
		if k_idx == 0:
			print_ = False
		for topi, v_ in enumerate(v):
			# print(k, len(v_))
			# check the categories of the retrieved
			
			row.append(np.sum(all_labels
                              [desired_cat[k_idx]][v_])/ topks[topi]) #np.sum(all_labels[desired_cat[k_idx]])
		df_row = pd.DataFrame([row], columns= col_test)
		df_skews = df_skews.append(df_row, ignore_index=True)

	df_skews = df_skews.round(2)
	df_skews.to_csv(f"result_csv/{fname}_relevance.csv")
	print(df_skews)
	return df_skews

def check_relevance_coco(retrieved_cat, desired_cat):
	# print(len(retrieved_cat))
	num_correct = 0
	for r in retrieved_cat:

		if r[desired_cat] == 1:
			num_correct +=1 

	return num_correct/ len(retrieved_cat)

def run_relevance_coco(queries, protected_attribute, similarities, all_cat, desired_cat,  fname):

	topks = [20, 50, 70]
	query_dict = {}
	for  idx, k in enumerate(queries):

		query_dict[k] = similarities[idx]
		
	sorted_idx = {}
	for k in query_dict.keys():
		s = np.asarray(query_dict[k])
		ind_sorted = s.argsort()
		sorted_idx[k] = []
		for topk in topks:
			ind = ind_sorted[-topk:][::-1]
			sorted_idx[k].append(ind)

	col_test = ['Query', 'recall_top_20', 'recall_top_50', 'recall_top_70']

	df_skews = pd.DataFrame(columns = col_test)
	for k_idx, (k, v) in enumerate(sorted_idx.items()):
		
		row = [k]
		print_ = False
		if k_idx == 0:
			print_ = False
		for topi, v_ in enumerate(v):
			# print(k, len(v_))
			# check the categories of the retrieved
			
			row.append(check_relevance_coco(all_cat[v_],  desired_cat[k_idx]))
		df_row = pd.DataFrame([row], columns= col_test)
		df_skews = df_skews.append(df_row, ignore_index=True)

	df_skews = df_skews.round(2)
	df_skews.to_csv(f"../results_csv/{fname}_relevance.csv")
	print(df_skews)
	return df_skews

def get_features_CelebA(dataset, model, device):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
            
            features = model.encode_image(images.to(device))
            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features), torch.cat(all_labels).cpu().numpy()

def get_CelebA(split,model, preprocess, device):
    import copy
    assert split in ["train", "test", "val", "all"], "split must be either 'train', 'test', 'val' or 'all' for CelebA"
    if split=='val':
        split = 'valid' 
    
    from torchvision.datasets import CelebA

	#     root = os.path.expanduser("~/efs-clip-experiments")
    CelebA_ds = CelebA("../../celebA", split=split, transform=preprocess, download=True)
    CelebA_features, CelebA_labels = get_features_CelebA(CelebA_ds, model, device)
    # normalizing features
    CelebA_features /= CelebA_features.norm(dim=-1, keepdim=True)
    #we use 'big lips' and 'black hair' as a proxy for race
    CelebA_labels = {
	#                     'arched_eyebrow': CelebA_labels[:,1],
	#                     'baggy_eyes' : CelebA_labels[:,3], 
	#                     'big_lips': CelebA_labels[:,6], 
	#                     'big_nose': CelebA_labels[:,7],
                    'black_hair': CelebA_labels[:,8],
                    'blond_hair': CelebA_labels[:,9],
                    'brown_hair' : CelebA_labels[:,11],
	#                     'bushy_eyebrows' : CelebA_labels[:,12],
	#                     'chubby' : CelebA_labels[:,13],
	#                     'double_chin' : CelebA_labels[:,14],
                    'glasses' : CelebA_labels[:,15],
	#                     'high_Cheekbones' : CelebA_labels[:,19],
                    'gender': CelebA_labels[:,20],
                    #'race': 1 * np.logical_and(CelebA_labels[:,6]==1, CelebA_labels[:,8]==1),
	#                     'oval_face': CelebA_labels[:,25],
	#                     'pointy_nose': CelebA_labels[:,27], # remove this 
                    
                    'smiling': CelebA_labels[:,31],
                    # 'straight_hair': CelebA_labels[:,32],  
                    'wavy_hair': CelebA_labels[:,33],  
        
                    'earrings': CelebA_labels[:,34],  
                    'hat': CelebA_labels[:,35], 
                     
                    'necktie': CelebA_labels[:,38],
                    'necklace': CelebA_labels[:,37],
                     
                    }
    CelebA_attr_to_int_dict = {'gender': {'female': 0, 'male': 1}, 
                              # 'race': {'not-black': 0, 'black': 1}
                              }
    CelebA_int_to_attr_dict = {'gender': {0: 'female', 1: 'male'}, 
                               #'race': {0: 'not-black', 1: 'black'}
                              }
    
    group_sizes = copy.deepcopy(CelebA_attr_to_int_dict)
    for attr in group_sizes.keys():
        for group_name, group_val in group_sizes[attr].items():
            group_sizes[attr][group_name] = np.sum(CelebA_labels[attr]==group_val)
            
    CelebA_ = {
        'features': CelebA_features,
        'labels': CelebA_labels,
        'int_to_attr': CelebA_int_to_attr_dict,
        'attr_to_int': CelebA_attr_to_int_dict,
        'nr_groups_to_consider': {'gender': 2, 
                                 # 'race': 2,
                                 },
        'group_sizes': group_sizes 
    }
    return CelebA_

def calculate_projections_celeba(clip_model, preprocess, device):
	
	import sys
	sys.path.insert(0, '../FairFace/')
	import fair_face_dataset as ff  
	from fair_pca_multi_group import apply_fair_PCA_to_dataset_multiple_groups

	data = get_CelebA("train", clip_model, preprocess, device)

	
	all_features_train, all_labels_gender_train  = data['features'], data['labels']['gender']

	all_labels_train = {'gender': all_labels_gender_train}

	projection_pca_gt = {}
	projection_pca_inf = {}
	MI_gt_dict = {}
	MI_inf_dict = {}
	
	all_features_train /= all_features_train.norm(dim=-1, keepdim=True)

	for attr in ['gender']:
		
		all_feat_labels = all_labels_train[attr]


		MI_GT = calc_mutual_info(all_features_train, all_feat_labels)
		# print("norm shape: ", all_features.norm(dim=-1, keepdim=True).shape)
		# later do inferred from clip model
		# idxs = np.where(all_feat_labels!= 2)[0]
		# print("dim fpca: ", all_features.shape, all_features.shape[1]-1)
		MI_gt_dict[attr] = MI_GT

		if len(np.unique(all_feat_labels)) > 2: 
			print(" unique attr", len(np.unique(all_feat_labels)))

			pipe_train = apply_fair_PCA_to_dataset_multiple_groups((all_features_train.cpu().numpy().astype(np.float64), all_labels_train['gender'], all_feat_labels),
				all_features_train.shape[1]-len(np.unique(all_feat_labels)) + 1, 
				LogisticRegression,
				'selection_rate_parity', 0, 
				standardize=False, fit_classifier=False)
		else:
			pipe_train = apply_fair_PCA_to_dataset((all_features_train.cpu().numpy().astype(np.float64), all_feat_labels, all_feat_labels),
				all_features_train.shape[1]- 1, 
				LogisticRegression,
				'selection_rate_parity', 0, 
				standardize=False, fit_classifier=False)

		projection_pca_gt[attr] = pipe_train

		inferred_att = infer_feat_ff(attr, all_features_train, all_labels_train, clip_model, device)

		MI_INF = calc_mutual_info(all_features_train, inferred_att)
		MI_inf_dict[attr] = MI_INF

		if len(np.unique(inferred_att)) > 2: 
			pipe_train_inferred = apply_fair_PCA_to_dataset_multiple_groups((all_features_train.cpu().numpy().astype(np.float64), all_labels_train['gender'], inferred_att),
				all_features_train.shape[1]- len(np.unique(all_feat_labels)) + 1, 
				LogisticRegression,
				'selection_rate_parity', 0, 
				standardize=False, fit_classifier=False)
		else:
			pipe_train_inferred = apply_fair_PCA_to_dataset((all_features_train.cpu().numpy().astype(np.float64), inferred_att, inferred_att),
			all_features_train.shape[1]-1, 
			LogisticRegression,
			'selection_rate_parity', 0, 
		standardize=False, fit_classifier=False)

		projection_pca_inf[attr] = pipe_train_inferred

	return projection_pca_gt, projection_pca_inf, MI_gt_dict, MI_inf_dict