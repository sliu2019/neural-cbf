import numpy as np
from cvxopt import solvers
solvers.options['show_progress'] = False
import pickle


def debug_format_1(fldrnm):
	# rollouts = pickle.load(open("rollout_results/flying/ours//rollouts.pkl", "rb"))
	rollouts = pickle.load(open("rollout_results/flying/ours/%s/rollouts.pkl" % fldrnm, "rb"))

	slack = rollouts["qp_slack"]
	eps_bdry = 1.0 # TODO: hardcoded
	eps_outside = 5.0 # TODO: hardcoded


	tminusone_slack = [rl[-2] for rl in slack]
	tminusone_slack = [x.item() for x in tminusone_slack if x is not None]
	tminusone_slack = np.array(tminusone_slack) - eps_bdry

	t_slack = [rl[-1] for rl in slack]
	t_slack = [x.item() for x in t_slack if x is not None]
	t_slack = np.array(t_slack) - eps_outside

	print(fldrnm)

	print("Avg t-1 slack: %.4f" % np.mean(tminusone_slack))
	print("Avg t slack: %.4f" % np.mean(t_slack))

	print("Max t-1 slack: %.4f" % np.max(tminusone_slack))
	print("Max t slack: %.4f" % np.max(t_slack))
	# IPython.embed()


if __name__ == "__main__":

	fldrnm = "exp_flying_inv_pend_phi_format_0_seed_0_ckpt_3370_nrollout_1000_dt_1.00E-04"
	fldrnm = "exp_flying_inv_pend_phi_format_1_seed_0_ckpt_60_nrollout_1000_dt_1.00E-04"
	debug_format_1(fldrnm)
	# import IPython
	# d = pickle.load(open("log/flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0/default_exp_data.pkl", "rb"))
	# IPython.embed()