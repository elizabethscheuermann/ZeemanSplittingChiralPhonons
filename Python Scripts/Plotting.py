import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse


# Arg Parse
parser = argparse.ArgumentParser()

parser.add_argument('-d', '--datasets', nargs="+", type = str)
parser.add_argument('-IV1', '--independent_variable1', type = str)
parser.add_argument('-IV2', '--independent_variable2', type = str)
parser.add_argument('-DV', '--dependent_variable', type = str)

parser.add_argument('-j', '--interaction_strength_index', type = int, default = 0)
parser.add_argument('-b', '--magnetic_field_index', type = int, default = 0)
parser.add_argument('-t', '--temperature_index', type = int, default = 0)

args = parser.parse_args()

three_level_filepath = "/home/twiggy/Research/Chiral_Phonon_Zeeman_Splitting/Data/ThreeLevel_Thermodynamics.csv"
four_level_filepath = "/home/twiggy/Research/Chiral_Phonon_Zeeman_Splitting/Data/FourLevel_Thermodynamics.csv"
six_level_filepath = "/home/twiggy/Research/Chiral_Phonon_Zeeman_Splitting/Data/SixLevel_Thermodynamics.csv"
filepaths = {"Triagonal": three_level_filepath,
             "Square": four_level_filepath,
             "Cubic": six_level_filepath}
plotlabels = {"B": "Magnetic Field Strength [B]",
              "J": "Interaction Strength [J]",
              "T": "Temperature [T]",
              "chi": r"Magnetic Susceptibility [$\chi$]"
              }
def main():
    # Get filepath
    datasets = args.datasets
    data = {}
    for dataset in datasets:
        data[dataset] = pd.read_csv(filepaths[dataset], delimiter = ', ')

    # Check independent and dependent variables
    if args.independent_variable1 not in ['J', 'B', 'T']:
        print("Invalid Independent Variable 1")
        return

    if args.independent_variable2 not in ['J', 'B', 'T', None]:
        print("Invalid Independent Variable 2")
        return

    if args.independent_variable1 == args.independent_variable2:
        print("Independent Var. 2 cannot be the same as Independet Var. 1")
        return

    if args.dependent_variable not in ['Ox', 'Oy', 'Oz', 'm', 'chi']:
        print("Invalid Dependent Variable")
        return

    # Get IVs and slices
    # Fixed IVS are those that are plotted against
    #
    SLICE_INDICES = {'J': args.interaction_strength_index, 'B': args.magnetic_field_index, 'T': args.temperature_index}
    SLICES = ['J', 'B', 'T']
    SLICES.remove(args.independent_variable1)
    IVS = [args.independent_variable1]

    if args.independent_variable2 != None:
        SLICES.remove(args.independent_variable2)
        IVS.append(args.independent_variable2)

    
    # Make figure
    fig, ax = plt.subplots()
    ax.set_xlabel(plotlabels[IVS[0]])
    if len(IVS) == 1:
        ax.set_ylabel(plotlabels[args.dependent_variable])
    elif len(IVS) == 2:
        ax.set_ylabel(plotlabels[IVS[1]])
    
    # Get Domains and data slices
    for dataset in datasets:
        data_set = data[dataset]
        slice_domains = {}
        for slice in SLICES: 
            slice_domains[slice] = np.unique(data_set[slice])


        IV_domains = {}
        for IV in IVS:
            IV_domains[IV] = np.unique(data_set[IV])


        z = data_set
        for slice in SLICES:
            z = z[z[slice] == slice_domains[slice][SLICE_INDICES[slice]]]
    
        z = np.array(z[args.dependent_variable]) 

        # Reshape data if needed
        if len(IVS) == 2:
            z = z.reshape((len(IV_domains[IVS[0]]),len(IV_domains[IVS[1]]))).T

        # Make figure
        if len(IVS) == 1:
            ax.plot(IV_domains[IVS[0]], z, label = dataset)

        if len(IVS) == 2:
            mesh_dom1, mesh_dom2 = np.meshgrid(IV_domains[IVS[0]], IV_domains[IVS[1]])
            p = ax.contourf(mesh_dom1, mesh_dom2, z)
            cbar = fig.colorbar(p)
        print(z)
    if (len(IVS) == 1): 
        ax.legend()
    plt.show()
main()
