#!/usr/bin/env python
"""
VISUALISE THE LIDAR DATA FROM THE ANY DATASET

Based on the sample code from
    https://github.com/utiasSTARS/pykitti/blob/master/demos/demo_raw.py
And:
    http://stackoverflow.com/a/37863912
    
Author: Gerald Baulig
"""

from mayavi import mlab
from mayavi.core.ui.api import MlabSceneModel, SceneEditor
from traits.api import HasTraits
from traitsui.api import View, Item, Group

class GUI(HasTraits):
    def __init__(self, items=None):
        self.fig = Instance(MlabSceneModel, ())
        self.view = View(
            Group(
                Item('scene1',
                    editor=SceneEditor(), height=250,
                    width=300),
                'button1',
                show_labels=False
                ),
            resizable=True
            )


def create_figure():
    return mlab.figure(bgcolor=(1, 1, 1), size=(640, 360))


def clear_figure(fig):
    mlab.clf(fig)


def mesh(X, T, Y, fig, plot=None):
    if not len(X):
        raise ValueError("Error: Empty frame!")

    if plot == None:
        plot = mlab.triangular_mesh(
            X[:,0],
            X[:,1],
            X[:,2],
            T,
            scalars=Y,
            colormap='spectral',  # 'bone', 'copper',
            line_width=10,        # Scale of the line, if any
            figure=fig,
            )
    else:
        plot.mlab_source.reset(
            x = X[:,0],
            y = X[:,1],
            z = X[:,2],
            triangles = T,
            scalars = Y
        )
    return plot


def normals(X, T, norms):
    if norms == None:
        norms = mlab.quiver3d(
            X[:,0],
            X[:,1],
            X[:,2],
            T[:,0],
            T[:,1],
            T[:,2],
            scale_factor=1.0)
    else:
        norms.mlab_source.reset(
            x = X[:,0],
            y = X[:,1],
            z = X[:,2],
            u = T[:,0],
            v = T[:,1],
            w = T[:,2],
        )
    return norms

def vertices(X, Y, fig, plot=None, comap='spectral', mode='point'):
    if not len(X):
        raise ValueError("Error: Empty frame!")
    
    if not isinstance(Y, np.ndarray):
        Y = np.zeros(X[:,0].shape) + Y

    if plot == None:
        plot = mlab.points3d(
            X[:,0],
            X[:,1],
            X[:,2],
            Y,
            mode=mode,         # How to render each point {'point', 'sphere' , 'cube' }
            colormap=comap,       # 'bone', 'copper',
            scale_factor=0.065,     # scale of the points
            scale_mode='none',
            line_width=1,        # Scale of the line, if any
            figure=fig,
            )
    else:
        plot.mlab_source.reset(
            x = X[:, 0],
            y = X[:, 1],
            z = X[:, 2],
            scalars = Y
        )
    return plot


if __name__ == '__main__':
    #Standard libs
    import sys
    from argparse import ArgumentParser

    #3rd-Party libs
    import numpy as np
    import pykitti

    #Local libs
    from utils import *

    def init_argparse(parents=[]):
        ''' init_argparse(parents=[]) -> parser
        Initialize an ArgumentParser for this module.
        
        Args:
            parents: A list of ArgumentParsers of other scripts, if there are any.
            
        Returns:
            parser: The ArgumentParsers.
        '''
        parser = ArgumentParser(
            description="Viz flownet3D output",
            parents=parents
            )
        
        parser.add_argument(
            '--input', '-X',
            metavar='WILDCARD',
            help="Wildcard to the ground truth files.",
            default='x_*.npy'
            )
        
        parser.add_argument(
            '--target', '-T',
            metavar='WILDCARD',
            help="Wildcard to the ground truth files.",
            default='t_*.npy'
            )
        
        parser.add_argument(
            '--pred', '-Y',
            metavar='WILDCARD',
            help="Wildcard to the prediction files.",
            default='y_*.npy'
            )
        
        parser.add_argument(
            '--sort', '-s',
            metavar='BOOL',
            nargs='?',
            type=bool,
            help="Sort the files?",
            default=False,
            const=True
            )
        
        return parser


    def validate(args):
        args.gt = myinput(
            "Wildcard to the input files.\n" + 
            "    input ('x_*.npy'): ",
            default='x_*.npy'
            )
        
        args.pred = myinput(
            "Wildcard to the target files.\n" + 
            "    target ('t_*.npy'): ",
            default='t_*.npy'
            )
        
        args.pred = myinput(
            "Wildcard to the prection files.\n" + 
            "    pred ('y_*.npy'): ",
            default='y_*.npy'
            )
        
        args.sort = myinput(
            "Sort the files?\n" + 
            "    sort (False): ",
            default=False
            )
        return args


    def main(args):
        # Load the data
        input_files = ifile(args.input, args.sort)
        target_files = ifile(args.target, args.sort)
        pred_files = ifile(args.pred, args.sort)
        fig = create_figure()
        plot = None
        norms = None

        for input_file, target_file, pred_file in zip(input_files, target_files, pred_files):
            input_batch = np.load(input_file)
            target_batch = np.load(target_file)
            pred_batch = np.load(pred_file)
            for gt, target, pred in zip(input_batch, target_batch, pred_batch):
                spl = int(gt.shape[0] / 2)
                data = np.concatenate((gt[:,:3], gt[:spl,:3] + pred), axis=0)
                scalar = np.zeros(data.shape[0])
                #scalar[:spl] = 0.2
                scalar[spl:spl*2] = 0.05
                scalar[spl*2:spl*3] = 0.8
                #scalar[spl*3:spl*4] = 0.8
                scalar[0] = 0.0
                scalar[1] = 1.0
                
                print(np.var(target, axis=0))
                
                plot = vertices(data, scalar, fig, plot, mode='sphere')
                norms = normals(gt[:spl,:3], pred, norms)
                fig.render() 
                myinput("Press to continue.")

        animator = animation()
        mlab.show(True)
        myinput("Press any key to quit:")
        return 0
    
    parser = init_argparse()
    args, _ = parser.parse_known_args()
    if len(sys.argv) == 1:
        args = validate(args)
    exit(main(args))