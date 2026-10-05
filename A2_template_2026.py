"""EC A2 template code - neuroevolution for targeted locomotion with ARIEL.

WHAT THIS FILE IS
-----------------
A *demo*, not a solution. It spawns a robot, drives it with a neural network
whose weights are RANDOM, runs the simulation, and reports how close the robot
ended up to a target.

There is deliberately NO evolution in here. Building the EA (representation,
initialisation, parent selection, variation, survivor selection) is the assignment.
See "YOUR JOB" at the bottom of this file.

THE ASSIGNMENT IN A NUTSHELL
------------------------------
Evolve the weights of a neural network controller so that a robot moves from
SPAWN_POS to TARGET_POSITION within the simulation time.

    fitness = distance between the robot's final position and TARGET_POSITION

HOW TO RUN
----------

Change MODE below to switch between an interactive viewer, a headless run,
a rendered video, or a single frame.
"""
import copy
import random
# Standard library
from pathlib import Path
from typing import Literal

# Third-party libraries
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer

# Local libraries (ARIEL)
from ariel import console
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots import john_set
from ariel.ec import set_seed
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.video_recorder import VideoRecorder

import ariel.simulation.tasks.targeted_locomotion as tl

from ariel.ec import (
    Individual,
    Population,
)

# Type aliases
type ViewerTypes = Literal["launcher", "video", "simple", "frame", "no_control"]

# --- RANDOM GENERATOR SETUP --- #
# Fix the seed while you are debugging.
# Report results over MULTIPLE seeds.
SEED = 42
RNG = np.random.default_rng(SEED)

# ariel.ec's own generators/mutators/crossover draw from a separate,
# package-level RNG. Reseed it too if you build your EA on ariel.ec,
# or every one of your "multiple seeds" runs the same variation operators.
set_seed(SEED)

# --- DATA SETUP --- #
SCRIPT_NAME = Path(__file__).stem
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

# --- EXPERIMENT CONSTANTS --- #
SPAWN_POS: list[float] = [0.0, 0.0, 0.1]  # where the robot starts
#ToDo discuss whether we want changeing target positions
# I think it would be unnecisarry work and difficult to explain why we chose it in the paper
TARGET_POSITION: list[float] = [2.0, 0.0, 0.1]  # where it should end up
SIM_DURATION: float = 10.0  # seconds of simulated time per evaluation
MODE: ViewerTypes = "launcher"  # see run_experiment() for the options


# ============================================================================ #
#  1. THE BODY AND THE WORLD
# ============================================================================ #
def build_world() -> SimpleFlatWorld:
    """Create the environment the robot lives in.

    YOU MAY CHANGE THIS. Options include: SimpleFlatWorld, RuggedTerrainWorld,
    CraterTerrainWorld, AmphitheatreTerrainWorld, OlympicArena, ...
    (SimpleTiltedWorld is not supported for this task.)

    Whatever you pick, keep it FIXED for all runs you compare against each
    other, and say in your report which one you used. A controller evolved on
    flat ground and one evolved on rugged terrain are not comparable numbers.
    """
    return SimpleFlatWorld()


def build_robot() -> CoreModule:
    """Create the robot body.

    YOU MAY CHANGE THIS. Options include the prebuilt bodies in
    `ariel.body_phenotypes.robogen_lite.prebuilt_robots` (gecko, spider, ...).

    Two consequences of this choice, and they matter:
      * The body determines `model.nu` (the number of hinges you must send
        commands to) - that is the OUTPUT size of your controller.
      * The body determines the size of `data.qpos` - if you feed qpos to your
        network, that is (part of) your INPUT size.
    Change the body and your genotype length changes with it. Keep the body
    FIXED within an experiment.
    """
    return john_set.gecko()


# ============================================================================ #
#  2. THE CONTROLLER CONTRACT
# ============================================================================ #
#
# MuJoCo calls the controller every physics step with (model, data); its job
# is to write into data.ctrl.
#
#   INPUTS   : whatever you read from `data` (qpos, qvel, time, ...), plus any
#              task info you already know, e.g. the vector to TARGET_POSITION.
#              INPUT SIZE is your choice, but must stay CONSTANT.
#   OUTPUTS  : exactly `model.nu` values, one per actuated hinge.
#   RANGE    : hinges accept [-pi/2, +pi/2] radians. A tanh output gives
#              [-1, 1] - rescale: actions * (np.pi / 2).
#   WRITING  : DIRECT (data.ctrl[:] = actions) commands the angle straight -
#              fast, but can destabilise the sim on large jumps. DELTA
#              (data.ctrl[:] += actions * alpha, alpha ~ 0.05, then clip) is
#              smoother but accumulates, so clipping is required. Pick one,
#              justify it, use it everywhere.
#   NaN      : blown-up weights silently write NaN into data.ctrl. Assert
#              against it while developing.
#
# ============================================================================ #

# Controller architecture - decide before writing your EA.
HIDDEN_SIZE: int = 6

def get_core_position(data: mj.MjData) -> npt.NDArray[np.float64]:
    """Return the robot core's current (x, y, z) world position."""
    return np.asarray(data.qpos[0:3]).copy()

def get_direction_vector(data: mj.MjData) -> npt.NDArray[np.float64]:
    """Return the vector that determines the direction from the
    Robots core to the target."""
    return  np.asarray(TARGET_POSITION) - np.asarray(data.qpos[0:3])

def nn_input(data):
    """Returns the input for the nn"""
    #ToDo talk with everybody about wether the input is good
    input = []
    # We input the distance to the target
    input.append(tl.distance_to_target(np.asarray(data.qpos[0:3]), np.asarray(TARGET_POSITION)))
    # We input the direction to the target
    input.extend(get_direction_vector(data))
    # We input the hinge angles
    input.extend(data.qpos[7:])
    # We input the hinge veloities
    input.extend(data.qvel[6:])
    #ToDo, might be interesting to add some "clock"
    # mechanism to the input?
    input.extend([np.sin(data.time * 2.0)])
    return input


def nn_controller(
    model: mj.MjModel,
    data: mj.MjData,
    weights: list[npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """Map robot state to hinge commands: in -> hidden -> actions.

    In this demo `weights` is drawn at RANDOM. In your assignment, `weights`
    is what the evolutionary algorithm produces: an individual's genotype,
    reshaped into these matrices. You are free to change the architecture
    itself (layers, activations, ...) - just keep input/output sizes correct.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model. Use `model.nu` for the number of hinges.
    data : mj.MjData
        The MuJoCo data. This is where you read the robot's state from.
    weights : list of ndarray
        [w1, w2] - the layer weight matrices.

    Returns
    -------
    npt.NDArray[np.float64]
        `model.nu` action values, already scaled to [-pi/2, pi/2].
    """
    w1, w2 = weights

    # --- INPUTS ---------------------------------------------------------- #
    # We apply the direction of the target as the input data.
    inputs = nn_input(data)

    # --- FORWARD PASS ----------------------------------------------------- #
    layer1 = np.tanh(inputs @ w1)
    outputs = np.tanh(layer1 @ w2)  # in [-1, 1]

    # --- RESCALE TO THE HINGE RANGE --------------------------------------- #
    return outputs * (np.pi / 2)  # in [-pi/2, pi/2]


def make_random_weights(
    input_size: int,
    output_size: int,
) -> list[npt.NDArray[np.float64]]:
    """Draw a random parameter set for `nn_controller`.

    THIS IS THE FUNCTION YOUR EA REPLACES. Instead of sampling weights from a
    normal distribution, your EA will search for them.

    Note the total parameter count printed by main(): that is the length of the
    flat vector an individual's genotype has to encode. Reshaping a flat
    genotype back into these matrices is on you.
    """
    return [
        RNG.normal(scale=0.5, size=(input_size, HIDDEN_SIZE)),
        RNG.normal(scale=0.5, size=(HIDDEN_SIZE, output_size)),
    ]

"""
Fitness functions
"""

FITNESS_FUNCTIONS = {
    "efficiency": tl.fitness_distance_and_efficiency,
    "locomotion": tl.fitness_survival_and_locomotion,
    "direct": tl.fitness_direct_path,
    "speed": tl.fitness_speed_to_target
}

def fitness_function(
    initial_position: npt.NDArray[np.float64],
    final_position: npt.NDArray[np.float64],
    min_z_position
) -> float:
    target = np.asarray(TARGET_POSITION)
    fitness = 0
    fitness += tl.fitness_delta_distance(initial_position, final_position, target)
    return fitness

"""
Evaluation
"""

def evaluate_individual(weights: np.ndarray) -> (np.ndarray, float):
    # MuJoCo's control callback is a GLOBAL. Clear it. DO NOT REMOVE.
    mj.set_mjcb_control(None)

    # --- World and robot --------------------------------------------------- #
    world = build_world()
    robot = build_robot()

    world.spawn(
        robot.spec,
        position=SPAWN_POS,
        correct_collision_with_floor=True,
    )

    # Compile the world into a model. USE AS IS.
    model = world.spec.compile()
    data = mj.MjData(model)

    # Put the simulation in a clean, known state before reading anything.
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    # When weights are not yet defined (initialisation)
    # we create random weights
    if weights is None:
        # --- Wire up the controller -------------------------------------------- #
        # We only perform this when the individual currently has no weights.
        # This is only the case for initialization
        # Sizes are read from the compiled model, never hardcoded - they depend on
        # the body you chose in build_robot()
        input_example = nn_input(data)
        input_size = len(input_example)
        output_size = model.nu

        weights = make_random_weights(input_size, output_size)

    def control_callback(m: mj.MjModel, d: mj.MjData) -> None:
        """Compute and apply actions; MuJoCo calls this every physics step."""
        actions = nn_controller(m, d, weights)

        # DIRECT application (see the controller contract above).
        d.ctrl[:] = actions

        # DELTA application - comment out the line above and use these instead:
        # delta = 0.05
        # d.ctrl[:] += actions * delta
        # d.ctrl[:] = np.clip(d.ctrl, -np.pi / 2, np.pi / 2)

    # --- Record the starting point ----------------------------------------- #
    initial_position = get_core_position(data)

    mj.set_mjcb_control(control_callback)   # register BEFORE stepping

    sim_steps = int(SIM_DURATION / model.opt.timestep)
    min_z = np.inf
    for _ in range(sim_steps):
        mj.mj_step(model, data)
        min_z = min(min_z, data.qpos[2])

    mj.set_mjcb_control(None)
    # --- Score -------------------------------------------------------------- #
    final_position = get_core_position(data)
    min_z_position = min_z
    fitness = fitness_function(initial_position, final_position, min_z_position)
    return weights, fitness
"""
Initialization
"""

def initialize_population(population_size:int=100) -> Population:
    # We randomly intitialize n bodies
    population = []
    for _ in range(population_size):
        # For this genome, we create a individual
        ind = Individual()
        # We randomly create weights and evaluate their fitness
        weights, fitness = evaluate_individual(None)
        # We assign these to the individual
        ind.genotype = weights
        ind.fitness = fitness
        # We add the individual to the population
        population.append(ind)
    # We then create a population object and return it
    return Population(population)

"""
Evolutionary steps
"""

def crossover(weights):
    # 1 point crossover for 2 parents
    # We deepcopy the weights
    weights = [copy.deepcopy(w) for w in weights]
    # We then perform the crossover for the weights
    w1p1, w2p1 = weights[0]
    w1p2, w2p2 = weights[1]
    cut_1 = random.randint(1, len(w1p1) - 2)
    cut_2 = random.randint(1, len(w2p1) - 2)
    # And perform crossover
    w1p1[:cut_1] = w1p2[:cut_1]
    w2p1[:cut_2] = w2p2[:cut_2]
    w1p2[:cut_1] = w1p1[:cut_1]
    w2p2[:cut_2] = w2p1[:cut_2]
    return weights

def mutate(weights, sigma: float = 0.2, rate: float = 0.1):
    """Gaussian mutation: each weight is perturbed with probability `rate`.

    Returns a new list of arrays; the input is left untouched.
    """
    mutated = []
    for w in weights:
        mask = RNG.random(w.shape) < rate
        noise = RNG.normal(0.0, sigma, size=w.shape)
        mutated.append(w + mask * noise)
    return mutated


def reproduction(population: Population, children_fraction: float = 0.25, tournament_size:int = 10, parent_amount:int =2) -> Population:
    # We determine how many children we want in our population
    n_children = round(children_fraction * len(population))
    # We perform tournament selection this amount of times
    for _ in range(n_children):
        # We take some random sample from the population
        tournament = population.sample(tournament_size)
        # We find the fittest in the population
        parents = tournament.best(sort = "min", n=parent_amount)
        parents = parents.to_list()
        # We set the weights and fitness for all parents in an array
        weights = [p.genotype for p in parents]
        # We perform crossover to the parents
        weights_children = crossover(weights)
        # We create individuals using these weights and fitnesses
        for w in weights_children:
            individual = Individual()
            mutated_weight = mutate(w)
            w_out, fitness = evaluate_individual(mutated_weight)
            individual.genotype = w_out
            individual.fitness = fitness
            # We add the children to the population
            population.append(individual)

    # We then kill of the individuals with the worst fitness
    worst = population.best(sort = "max", n=int(n_children*parent_amount))
    for ind in worst:
        ind.alive = False
    return population.alive

"""
Launch Ariel
"""

def show_behavior(individual: Individual, filename = "test"):
    world = build_world()
    robot = build_robot()

    world.spawn(
        robot.spec,
        position=SPAWN_POS,
        correct_collision_with_floor=True,
    )

    # Compile the world into a model. USE AS IS.
    model = world.spec.compile()
    data = mj.MjData(model)

    # Put the simulation in a clean, known state before reading anything.
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    weights = individual.genotype

    def control_callback(m: mj.MjModel, d: mj.MjData) -> None:
        """Compute and apply actions; MuJoCo calls this every physics step."""
        actions = nn_controller(m, d, weights)
    #ToDo this is just a random crossover implementation for testing
    # It splits the two weight vectors in two and performs crossover

        # DIRECT application (see the controller contract above).
        d.ctrl[:] = actions

        # DELTA application - comment out the line above and use these instead:
        # delta = 0.05
        # d.ctrl[:] += actions * delta
        # d.ctrl[:] = np.clip(d.ctrl, -np.pi / 2, np.pi / 2)

    mj.set_mjcb_control(control_callback)
    recorder = VideoRecorder(file_name=filename, output_folder=str(DATA / "__videos__"))
    video_renderer(
        model,
        data,
        duration=SIM_DURATION,
        video_recorder=recorder,
    )
    mj.set_mjcb_control(None)

if __name__ == "__main__":
    pop = initialize_population(50)
    best = pop.best(sort="min", n=1)[0]
    print(best.fitness)
    show_behavior(best, "start")
    for _ in range(50):
        pop = reproduction(pop)
        best = pop.best(sort="min", n=1)[0]
        print(best.fitness)
    show_behavior(best, "finish")


# ============================================================================ #
#  YOUR JOB
# ============================================================================ #
#
# Everything above runs one robot with random weights. It will score badly, and
# it will score badly in a slightly different way every time you change SEED.
# Your task is to replace "random" with "evolved".
#
# Build a proper EA on top of `ariel.ec`. You are expected to use that module -
# it gives you the population/individual data model, the operators, and free
# persistence of every generation to a SQLite database, which you will want
# when it is time to plot convergence curves for the report.
#
#     from ariel.ec import EA, EAOperation, Individual, Population
#
# For a complete, runnable example of how those pieces fit together (a one-max
# EA with parent selection, crossover, mutation and survivor selection written
# as separate steps), read:
#
#     examples/new_EC_engine_example.py
#
# and the API documentation at:
#
#     https://ci-group.github.io/ariel/
#
# ---- EXPERIMENTAL RIGOUR ---------------------------------------------------
#
#   One run proves nothing. Repeat every configuration over several
#     independent seeds and report mean and spread.
#   Log best/mean/worst fitness per generation. The database `ariel.ec`
#     writes makes this straightforward.
#   Compare against a baseline. Random search with the same evaluation
#     budget is a simple, but reasonable choice; and it is nearly free to run.
#   Keep body, world, SIM_DURATION and fitness function identical across
#     everything you compare. Change one thing at a time.
#
# ============================================================================ #
