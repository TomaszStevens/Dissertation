import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import string
from matplotlib.pyplot import figure
import math
from fractions import Fraction

def horizontal_plot(path, file, colour_scheme=2, custom_title=None, maxAlphaExponent=1, minAlphaExponent=-1):

    if path not in ['A','B']:
        raise Exception('Path must be equal to either "A" or "B"')

    df = pd.read_csv(file)

    fig, ax = plt.subplots(figsize=(13.1, 6))

    # Palette for plot
    palette = [sns.cubehelix_palette(df.shape[1], start=.5, rot=-.75)]
    palette.append(sns.cubehelix_palette(df.shape[1]))

    palette = palette[colour_scheme-1]

    # Find sum for each row
    totals = df.sum(axis=1).to_list()

    # Convert column values to percentages for each row
    bars = []
    for col in df:
        bars.append([i/j * 100 for i,j in zip(df[col], totals)])

    # Create plot
    r = range(df.shape[0])
    barWidth = 0.85
    breadth = (df.shape[0])//2
    if path == 'A' and breadth != 0:
        powerDiff = (maxAlphaExponent-minAlphaExponent)/(df.shape[0]-1)
        fractions = [Fraction(minAlphaExponent+powerDiff*i).limit_denominator() for i in range(0,df.shape[0])]
        max_denom = np.max([fraction.denominator for fraction in fractions])
        numerators = [int(fraction.numerator*(max_denom/fraction.denominator)) for fraction in fractions]
        names = [f'{num}/{max_denom}' if num/max_denom %1 != 0 else str(num/max_denom) for num in numerators]
    elif path == 'A':
        names = [0]
    elif df.shape[0] == 3:
        names = [1,0,-1]
    else:
        raise Exception('File does not match Path B - the number of rows does not equal 3')

#     names = [f'{i}/{int(breadth/boundary_exponent)}' if (boundary_exponent*i/breadth)%1 != 0 else str(boundary_exponent*i/breadth) for i in range(-breadth,breadth+1)]
    labels = ['1st Preference','2nd Preference','3rd Preference','4th Preference','5th Preference',
              '6th Preference','7th Preference','8th Preference','9th Preference','10th Preference',
             '11th Preference','12th Preference','13th Preference','14th Preference','15th Preference',
             '16th Preference','17th Preference','18th Preference','19th Preference','20th Preference'] + ['overflow']*df.shape[1]

    left = np.zeros(df.shape[0])
    for i in range(df.shape[1]):
        ax.barh(names, width=bars[i], left=left, color=palette[i], edgecolor='white', height=barWidth, label=labels[i])
        text_color = 'white' if np.prod(palette[i]) < 0.5 else 'black'
        for j in r:
            if bars[i][j] > 3:
                size='small' if bars[i][j] < 5 else 'medium'
                ax.text(left[j]+0.5*bars[i][j],names[j],str(round(bars[i][j],1))+'%', ha='center',
                        va='center', color=text_color, size=size)
        left = [a+b for a,b in zip(left, bars[i])]

    # Custom axis
    if path == 'A':
        ax.set_title("% of Students Given Their i'th Preference as a Project\n(For Different Values of "+r'$\alpha$'")" if not custom_title else custom_title,
                     size='xx-large', pad=35 + 15*(np.floor(df.shape[1]/6)))
        ax.set_ylabel("Value of " + r'$\alpha$ (exponents of n)', size='x-large')

    elif path == 'B':
        ax.set_title("% of Students Given Their i'th Preference as a Project\n(For Different Objective Functions)" if not custom_title else custom_title,
                     size='xx-large', pad=35 + 15*(np.floor(df.shape[1]/6)))
        plt.tick_params(axis='y' ,which='both', left=False, labelleft=False)
        fig.text(0.066, 0.683, 'Sequential\nMinimisation', ha='center', size='x-large')
        fig.text(0.066, 0.448, 'Middle-Ground\nApproach', ha='center', size='x-large')
        fig.text(0.066, 0.210, 'Sequential\nMaximisation', ha='center', size='x-large')

    ax.set_xlabel("Percentage of Students", size='x-large')

    ax.legend(ncol=df.shape[1] if df.shape[1] < 7 else int(np.ceil(df.shape[1]/(np.ceil(df.shape[1]/6)))), loc='center', bbox_to_anchor=(0.5, 1.05 + 0.025*(np.floor(df.shape[1]/6))), fontsize='medium')

    # Show plot
    plt.show()


# ====================================== READ ME ============================================= #


# Here, write the "Path" that your data file took within the optimisation solver (i.e. 'A' or 'B')
path = 'A'

# Here, write the path to the output datafile that you would like to visualise (string)
# file = 'example.csv'

file = 'out/PB3_output_breadth_5.csv'

# 'horizontal_plot()' takes 4 extra parameters (none are required, however, 'maxAlphaExponent' and 'minAlphaExponent'
#                                          should be used if using 'customAlphas' in FICO Xpress) ... these are :
# ... 'colour_scheme' = {1,2}
# ... 'custom_title' = string
# ... 'maxAlphaExponent' = float or int (enter here the value of maxAlphaExponent used in FICO Xpress)
# ... 'minAlphaExponent' = float or int (enter here the value of minAlphaExponent used in FICO Xpress)
horizontal_plot(path=path, file=file, colour_scheme=2, maxAlphaExponent = 5/5, minAlphaExponent = -5/5)
