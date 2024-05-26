import argparse
import os
import math


def parse_vmodel_args() -> argparse.Namespace:

    def formatter_class(prog):
        return argparse.ArgumentDefaultsHelpFormatter(prog,
                                                      max_help_position=52,
                                                      width=90)

    parser = argparse.ArgumentParser(description='vmodel',
                                     formatter_class=formatter_class)

    # Script arguments
    formats = ['netcdf', 'pickle']
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='verbose output')
    parser.add_argument('-f', '--file', type=str, default='',
                        help='read parameters from YAML file')
    parser.add_argument('-p', '--plot', action='store_true', default=False,
                        help='show live plots')
    parser.add_argument('--plot-metrics', action='store_true', default=False,
                        help='show live plots of metrics')
    parser.add_argument('--plot-every', type=int, default=10, metavar='K',
                        help='plot every k timesteps')
    parser.add_argument('--plot-blocking', action='store_true', default=False,
                        help='wait for key press to resume plotting')
    parser.add_argument('-j', '--jobs', type=int, default=os.cpu_count(),
                        metavar='J', help='number of parallel jobs')
    parser.add_argument('-n', '--dry-run', action='store_true', default=False,
                        help='dry run, do not save data')
    parser.add_argument('-P', '--progress', action='store_true', default=False,
                        help='show progress bar')
    parser.add_argument('--save-every', type=int, default=1, metavar='K',
                        help='save data only every k timesteps')
    parser.add_argument('--parallel-agents', action='store_true', default=False,
                        help='process every agent in parallel')
    parser.add_argument('--no-parallel-runs', action='store_true',
                        default=False, help='do not process runs in parallel')
    parser.add_argument('--no-save-precomputed', action='store_true',
                        default=False,
                        help='save precomputed variables (saves memory)')
    parser.add_argument('--format', choices=formats, default='netcdf',
                        help='format for saved dataset')
    parser.add_argument('--no-compress', action='store_true', default=False,
                        help='do not compress datasets')
    parser.add_argument('--outfolder', default="",
                        help='output location')

    # Experimental arguments
    algorithms = ['reynolds', 'olfati']
    spawns = ['poisson', 'uniform', 'grid']
    experiment = parser.add_argument_group('experiment arguments')
    experiment.add_argument('--num-agents', type=int, default=10, metavar='N',
                            help='number of agents')
    experiment.add_argument('--num-runs', type=int, default=1, metavar='N',
                            help='number of runs')
    experiment.add_argument('--num-dims', type=int, default=2, metavar='N',
                            help='number of dimensions')
    experiment.add_argument('--delta-time', type=float, default=0.1,
                            metavar='SEC',
                            help='time delta between timesteps [s]')
    experiment.add_argument('--num-timesteps', type=int, default=None,
                            metavar='K',
                            help='number of timesteps for experiment')
    experiment.add_argument('--algorithm', choices=algorithms,
                            default='reynolds', help='flocking algorithm')
    experiment.add_argument('--spawn', choices=spawns, default='uniform',
                            help='spawn method')
    experiment.add_argument('--spawn-distance', type=float, default=None,
                            metavar='F', help='spawn distance')
    experiment.add_argument('--seed', type=int, default=None,
                            help='set seed for repeatability')
    #############################################################
    #new args
    #############################################################
    experiment.add_argument('--num-preds', type=int, default=5, metavar='NP',
                            help='number of predators')
    
    experiment.add_argument('--flee-range', type=float, default=10, metavar='NP',
                            help='range at which prey flees')
    
    experiment.add_argument('--flee-strength', type=float, default=5, metavar='NP',
                            help='magnitude of flee force')
    
    experiment.add_argument('--flee-ang', type=float, default=30, metavar='NP',
                            help='fleeing angle for prey')
    
    experiment.add_argument('--hunt-str', type=float, default=1, metavar='NP',
                            help='magnitude of hunt force')
    
    experiment.add_argument('--vision-prey', type=float, default=330, metavar='NP',
                            help='field of vision for prey')
    
    experiment.add_argument('--vision-pred', type=float, default=120, metavar='NP',
                            help='field of vision for predators')
    
    experiment.add_argument('--alignment-strength', type=float, default=3, metavar='NP',
                            help='alignment strength of preys')
    
    experiment.add_argument('--dphi', type=float, default=0.2, metavar='NP',
                            help='noise magnitude for movement')
    
    experiment.add_argument('--repulsion-prey', type=float, default=3, metavar='NP',
                            help='magnitude of prey-prey repulsion')
    
    experiment.add_argument('--attraction-prey', type=float, default=3, metavar='NP',
                            help='magnitude of prey-prey attraction')
    
    experiment.add_argument('--repulsion-pred', type=float, default=21, metavar='NP',
                            help='magnitude of pred-pred repulsion')
    
    experiment.add_argument('--repradius-prey', type=float, default=1.5, metavar='NP',
                            help='radius of prey-prey repulsion')
    
    experiment.add_argument('--attradius-prey', type=float, default=1.5, metavar='NP',
                            help='radius of prey-prey attraction')
    
    experiment.add_argument('--repradius-pred', type=float, default=20, metavar='NP',
                            help='radius of pred-pred num-ulsion')
    
    experiment.add_argument('--repulsion-col', type=float, default=10000000, metavar='NP',
                            help='magnitude of collision repulsion')
    
    experiment.add_argument('--prey-size', type=float, default=.25, metavar='NP',
                            help='radius of prey')
    
    experiment.add_argument('--pred-size', type=float, default=.25, metavar='NP',
                            help='radius of predator')

    experiment.add_argument('--pred-time', type=int, default=200, metavar='PT',
                            help='time predator appears')
    
    experiment.add_argument('--pred-hunt', type=int, default=1, metavar='PH',
                            help='behavior of predator, 1 = continous, 2 = one attack')
    
    experiment.add_argument('--pred-angle', type=int, default=180, metavar='PA',
                            help='angle relative to swarm velocity the predator gets initialized with, if negative, random angles are chosen')
    
    experiment.add_argument('--pred-dist', type=int, default=10, metavar='PD',
                            help='distance to COM of prey the predator gets initialized with')
    
    experiment.add_argument('--col-style', type=int, default=1, metavar='CS',
                            help='0: no collisions, 1: only react to closest collision, 2: account for multiple collisions')
    experiment.add_argument('--trans-pred', type=int, default=0, metavar='TP',
                            help='time frame after pred-time during which preds do not hunt to allow for lattice formation')
    
    experiment.add_argument('--pred-segments', type=int, default=0, metavar='PSG',
                            help='amount od predator segments, 0 for single particle predator')

    experiment.add_argument('--pred-length', type=int, default=3, metavar='PLN',
                            help='distance between predator segments')
    # Perception arguments
    perception = parser.add_argument_group('perception arguments')
    perception.add_argument('--radius', type=float, default=0.25, metavar='F',
                            help='radius of an agent [m]')
    perception.add_argument('--perception-radius', type=float, default=0.0,
                            metavar='F', help='perception radius of agents [m]')
    perception.add_argument('--perception-angle', type=float, default=0.0,
                            metavar='DEG',
                            help='angle below which objects are invisible [deg]')
    perception.add_argument('--filter-occluded', action='store_true',
                            default=False,
                            help='if true, filter out occluded agents')
    perception.add_argument('--filter-voronoi', action='store_true',
                            default=False,
                            help='if true, filter out non-voronoi neighbors')
    perception.add_argument('--max-agents', type=int, default=0, metavar='N',
                            help='maximum number of closest agents to consider')
    perception.add_argument('--false-negative-prob', type=float, default=0.0,
                            metavar='P',
                            help='false negative probability [prob]')
    perception.add_argument('--num-clutter', type=float, default=0.0,
                            metavar='K',
                            help='avg number of clutter returns per timestep')
    perception.add_argument('--range-std', type=float, default=0.0,
                            metavar='STD',
                            help='distance-scaled range std dev [m]')
    perception.add_argument('--bearing-std', type=float, default=0.0,
                            metavar='STD',
                            help='bearing std dev [deg]')
    perception.add_argument('--visual-migration', action='store_true',
                            default=False,
                            help='if true, subject waypoint to occlusion')
    perception.add_argument('--topo-angle', type=float, default=0.0,
                            metavar='DEG',
                            help='minimum angle between closest agents [deg]')

    # Control arguments
    control = parser.add_argument_group('control arguments')
    control.add_argument('--ref-distance', type=float, default=1,
                         metavar='F', help='desired inter-agent distance [m]')
    control.add_argument('--migration-gain', type=float, default=0.5,
                         metavar='K', help='migration gain [m/s]')
    control.add_argument('--migration-point', nargs=2, type=float, default=[0,0],
                         metavar=('X', 'Y'), help='migration point (x, y)')
    control.add_argument('--migration-dir', nargs=2, type=float, default=[],
                         metavar=('X', 'Y'), help='migration direction (x, y)')
    control.add_argument('--max-speed', type=float, default=1, metavar='F',
                         help='Maximum speed [m/s]')

    # Need to parse known args when launching from ROS node
    # Reason: __name and __log are not specified above
    args, _ = parser.parse_known_args()

    return args
