import matplotlib as mpl

def plot_params(fontsize=16,linewidth=1.5):
    """
    Procedure to set the parameters for this suite of plotting utilities
    """
    
    mpl.rcParams['axes.linewidth'] = linewidth
    mpl.rcParams['xtick.major.size'] = 5
    mpl.rcParams['xtick.major.width'] = linewidth
    mpl.rcParams['ytick.major.size'] = 5
    mpl.rcParams['ytick.major.width'] = linewidth
    mpl.rcParams['xtick.labelsize'] = fontsize
    mpl.rcParams['ytick.labelsize'] = fontsize

    return

def plot_defaults():
    mpl.rcdefaults()
    return
