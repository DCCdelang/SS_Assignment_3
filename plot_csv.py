import matplotlib.pyplot as plt
import pandas as pd

def two_opt_vs_SA():
    df_2opt = pd.read_csv("data/2-opt.csv")
    two_opt_iterations = df_2opt['iterations']
    two_opt_costs = df_2opt['costs']

    df_SA = pd.read_csv("data/0.975.csv")
    SA_iterations = df_SA['iterations']
    SA_costs = df_SA['costs']

    plt.plot(two_opt_iterations, two_opt_costs, label = '2-opt')
    plt.plot(SA_iterations, SA_costs, label = 'SA')

    plt.xlabel('MCMC (iterations)', fontsize=14)
    plt.ylabel('Cost', fontsize=14)
    plt.legend(fontsize=14)
    plt.tick_params(labelsize=14)
    plt.show()


two_opt_vs_SA()