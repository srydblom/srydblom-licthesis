
def setAxLinesBW(ax):
    """
    Take each Line2D in the axes, ax, and convert the line style to be
    suitable for black and white viewing.
    """
#    MARKERSIZE = 3

    COLORMAP = {
        0 : {'marker': None, 'dash': (None,None)},
        1 : {'marker': None, 'dash': [5,5]},
        2 : {'marker': None, 'dash': [5,3,1,3]},
        3 : {'marker': None, 'dash': [1,3]},
        4 : {'marker': None, 'dash': [5,2,5,2,5,10]},
        5 : {'marker': None, 'dash': [5,3,1,2,1,10]},

        #'k': {'marker': 'o', 'dash': (None,None)} #[1,2,1,10]}typ
        }
    count = 0
   # for index, key in enumerate(COLORMAP):
        #print index, key
    if ax.get_legend() != None:
        fig_lines = ax.get_lines() + ax.get_legend().get_lines()
        legend = True
    else:
        fig_lines = ax.get_lines()
        legend = False

    if legend:
        mod = len(fig_lines)/2
    else:
        mod = len(fig_lines)

    for line in fig_lines:
        print "setting line: ", line
        #origColor = line.get_color()
        line.set_color('black')
        line.set_dashes(COLORMAP[count % mod]['dash'])
        line.set_marker(COLORMAP[count % mod]['marker'])
        count+=1
        if count == 6:
            count = 0

def setFigLinesBW(fig):
    """
    Take each axes in the figure, and for each line in the axes, make the
    line viewable in black and white.
    """
    for ax in fig.get_axes():
        setAxLinesBW(ax)
