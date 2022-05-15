import pickle, os
import IPython

if __name__ == "__main__":

	fldrs = ["flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0", "flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0", "flying_inv_pend_ESG_reg_speedup_better_attacks_seed_1", "flying_inv_pend_ESG_reg_speedup_better_attacks_seed_2", "flying_inv_pend_ESG_reg_speedup_better_attacks_seed_3", "flying_inv_pend_ESG_reg_speedup_better_attacks_seed_4"]
	fnms = ["long_length_ckpt_1020_exp_data.pkl", "long_length_ckpt_250_exp_data.pkl", "long_length_ckpt_375_exp_data.pkl", "long_length_ckpt_625_exp_data.pkl", "long_length_ckpt_250_exp_data.pkl", "long_length_ckpt_175_exp_data.pkl"]

	rollout_keys_of_interest = ['percent_on_in', 'percent_on_out', 'percent_on_on', 'N_on_in_outside_box', 'N_on_out_outside_box', 'N_on_on_outside_box']
	for fldr, fnm in zip(fldrs, fnms):
		fpth = os.path.join("./log/", fldr, fnm)
		data = pickle.load(open(fpth, "rb"))

		print("************************************************************")
		print(fpth)
		# print(data.keys())
		"""
		'args', 'percent_infeasible', 'n_infeasible', 'mean_infeasible_amount', 'std_infeasible_amount', 'average_boundary_debug_dict', 'worst_infeasible_amount', 'worst_x', 'rollout_info_dicts', 'rollout_stat_dict', 'percent_of_domain_volume'
		
		Keys from rollout stat dict 
		dict_keys(['N_transitions', 'percent_on_in', 'percent_on_out', 'percent_on_on', 'N_on_in', 'N_on_out', 'N_on_on', 'min_phi', 'mean_phi', 'mean_dist', 'max_dist', 'mean_phi_grad', 'max_phi_grad', 'percent_on_in_outside_box', 'percent_on_out_outside_box', 'percent_on_on_outside_box', 'N_on_in_outside_box', 'N_on_out_outside_box', 'N_on_on_outside_box', 'N_count_exit_on_dalpha', 'mean_violation_amount', 'std_violation_amount'])
		"""

		args = data["args"]
		print("% infeas.: ", data["percent_infeasible"])
		print(data["mean_infeasible_amount"], " +/- ", data["std_infeasible_amount"])
		print("n_{boundary_samples}: ", args["boundary_n_samples"])
		print("\n")

		print("worst infeas. :", data["worst_infeasible_amount"])
		print("n_{boundary_samples}: ", args["worst_boundary_n_samples"])
		print("worst x:", data["worst_x"])
		print("\n")

		rollout_stat_dict = data["rollout_stat_dict"]
		for key in rollout_keys_of_interest:
			print(key, rollout_stat_dict[key])
		print("n_{rollouts}: ", args["rollout_N_rollout"])
		print("\n")

		print("volume: ", data["percent_of_domain_volume"])
		print("n_{domain_samples}: ", args["N_samp_volume"])
		# IPython.embed()

		print("************************************************************")
