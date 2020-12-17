import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

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


# two_opt_vs_SA()

def T_compare_plot(c=0.975):
    T_var = [1]
    c_var = [0.3,0.5,0.7,0.9]
    for c in c_var:
        for T in T_var:
            # print("succes",T,c)
            df_temp = pd.read_csv(f"data/two_opt_anneal/lin_c_{c}_temp_{T}.csv")
            means = df_temp['Means']
            # stds = df_temp['Std']
            # ci = (1.96*stds/means)
            plt.plot(range(len(means)), means, label = f'c = {c}')
            # plt.fill_between(range(len(means)),(means-ci),(means+ci),alpha=0.8)
            # print("For T =",T, "value =",means.iloc[-1],"+-",stds.iloc[-1])
            # print("For T =",T, "value =",means.iloc[-1])

    plt.xlabel('MCMC (iterations)', fontsize=14)
    plt.ylabel('Cost', fontsize=14)
    plt.legend(fontsize=14)
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig("figures/exp_c.png")
    plt.show()
T_compare_plot()


def function(data, a, b, c):
    x = data[0]
    y = data[1]
    return a * (x**b) * (y**c)

def T_c_compare_plot():
    T_var = [100,1000,5000,10000,20000]
    c_var = [0.3,0.5,0.7,0.9,0.975]
    # T_var = [100,1000,10000,50000,100000]
    # c_var = [0.8,0.9,0.99]
    x_data = []
    y_data = []
    z_data = []
    fig = plt.figure()
    ax = Axes3D(fig)
    for c in c_var:
        for T in T_var:
            df_temp = pd.read_csv(f"data/two_opt_anneal/lin_c_{c}_temp_{T}.csv")
            # df_temp = pd.read_csv(f"data/random_anneal/exp_c_{c}_temp_{T}.csv")
            means = df_temp['Means']
            x_data.append(T)
            y_data.append(c)
            z_data.append(means.iloc[-1])
            # print("For T =",T, "value =",means.iloc[-1])

    parameters, _ = curve_fit(function, [x_data, y_data], z_data)

    # create surface function model
    # setup data points for calculating surface model
    model_x_data = np.linspace(min(x_data), max(x_data), 30)
    model_y_data = np.linspace(min(y_data), max(y_data), 30)
    # create coordinate arrays for vectorized evaluations
    X, Y = np.meshgrid(model_x_data, model_y_data)
    # calculate Z coordinate array
    Z = function(np.array([X, Y]), *parameters)

    ax.plot_surface(X, Y, Z,alpha=0.5)
    ax.scatter(x_data,y_data,z_data,c="r")
    ax.set_xlabel('\nInitial T', fontsize=14)
    ax.set_xticks([0,5000,10000,15000,20000])
    ax.set_ylabel('\nc value', fontsize=14)
    ax.set_zlabel("\nCost",fontsize=14)
    ax.dist = 13
    # plt.legend(fontsize=14)
    plt.tick_params(labelsize=12)
    plt.savefig("figures/TOA_T_c_compare.png")
    plt.show()
# T_c_compare_plot()