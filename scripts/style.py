
import seaborn as sns
import numpy as np
from matplotlib import rc
import matplotlib as mpl
import matplotlib.pyplot as plt

# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# for Palatino and other serif fonts use:
sns.set_context(rc={'lines.markeredgewidth': 0.1})
sns.set_style("white", {'axes.grid': True,
                              'grid.linestyle': ':',
                              'xtick.major.size': 4,
                              'ytick.major.size': 4
                              })
rc('text', usetex=True)
#rc('font', **{'family': 'serif', 'serif': ['MinionPro'], 'size': 24})
rc('legend', **{'fontsize': 16})

#rc('mathtext', **{'fontset' : 'custom'})
#rc('font', family='serif', serif='Minion Pro')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('axes', titlesize=20)
rc('axes',labelsize='large')  # fontsize of the x any y labels
#rc('lines',linewidth=2)
#rc('text.latex', preamble=r'\usepackage{cmbright}')
#rc('axes',linewidth=1.5)
#rc('xtick.major',size=13.5,width=2)
#rc('ytick.major',size=13.5,width=2)
rc('text.latex', preamble=[
       r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
       r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
       r'\usepackage{MinionPro}',    # set the normal font here
       r'\usepackage{textgreek}'
])


OUTPATH = '/home/davkra/Documents/phdthesis/thesis/figures/'

#seaborn.set(style='white')
#seaborn.despine(top=True,right=True)


def fwhm(dat, normalize=False):
    fwhm = np.array(dat)*2*np.sqrt(2*np.log(2))
    if normalize:
        return fwhm/fwhm.max()
    else:
        return fwhm

class MyTransform(mpl.transforms.Transform):
    input_dims = 2
    output_dims = 2

    def __init__(self, base_point, base_transform, offset, *kargs, **kwargs):
        self.base_point = base_point
        self.base_transform = base_transform
        self.offset = offset
        super(mpl.transforms.Transform, self).__init__(*kargs, **kwargs)

    def transform_non_affine(self, values):
        new_base_point = self.base_transform.transform(self.base_point)
        t = mpl.transforms.Affine2D().translate(-new_base_point[0], -new_base_point[1])
        values = t.transform(values)
        x = values[:, 0:1]
        y = values[:, 1:2]
        r = np.sqrt(x**2+y**2)
        new_r = r-self.offset
        new_r[new_r<0] = 0.0
        new_x = new_r/r*x
        new_y = new_r/r*y
        return t.inverted().transform(np.concatenate((new_x, new_y), axis=1))


def my_plot(X, Y, linestyle='-', color='k', marker='o'):
    ax = plt.gca()
    line, = ax.plot(X, Y,color, marker=marker, linestyle='', fillstyle='none',mew=1, label='')
    color = line.get_color()

    size = X.size
    for i in range(1,size):
        mid_x = (X[i]+X[i-1])/2
        mid_y = (Y[i]+Y[i-1])/2

        # this transform takes data coords and returns display coords
        t = ax.transData

        # this transform takes display coords and
        # returns them shifted by `offset' towards `base_point'
        my_t = MyTransform(base_point=(mid_x, mid_y), base_transform=t, offset=7)

        # resulting combination of transforms
        t_end = t + my_t

        line, = ax.plot(
          [X[i-1], X[i]],
          [Y[i-1], Y[i]],
          linestyle=linestyle, color=color)
        line.set_transform(t_end)
    return line

# some useful color paletts
cold_hot = np.array([[5,   48,   97],
        [ 33,  102,  172],
        [ 67,  147,  195],
        [146,  197,  222],
        #[209,  229,  240],
        [247,  247,  247],
        #[254,  219,  199],
        [244,  165,  130],
        [214,   96,   77],
        [178,   24,   43],
        [103,    0,   31]])/256.
#coldhot = seaborn.blend_palette(cold_hot, 8)
white_red_blue=np.array([[254, 254, 254],
                 [254, 254, 160],
                 [254, 254,  99],
                 [244, 244, 110],
                 [255, 210,  35],
                 [255, 163,  25],
                 [255,  89,  25],
                 [230, 122, 101],
                 [237, 145, 124],
                 [239, 178, 146],
                 [247, 199, 178],
                 [255, 230, 230],
                 [215, 225, 255],
                 [150, 210, 255],
                 [ 30, 189, 255],
                 [ 20, 159, 255],
                 [ 10, 108, 240],
                 [ 11, 116, 255],
                 [ 10, 104, 200],
                 [  0,  89, 159]])/256.

paired = ((0.650980, 0.807843, 0.890196),
            (0.121569, 0.470588, 0.705882),
            (0.698039, 0.874510, 0.541176),
            (0.200000, 0.627451, 0.172549),
            (0.984314, 0.603922, 0.600000),
            (0.890196, 0.101961, 0.109804),
            (0.992157, 0.749020, 0.435294),
            (1.000000, 0.498039, 0.000000),
            (0.792157, 0.698039, 0.839216),
            (0.415686, 0.239216, 0.603922),
            (1.000000, 1.000000, 0.600000),
            (0.694118, 0.349020, 0.156863))

def color_palette(name, no):
    return sns.blend_palette(name, no)