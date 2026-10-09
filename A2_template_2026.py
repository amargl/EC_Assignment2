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
from crypt import methods
# Standard library
from pathlib import Path
from typing import Literal
import os

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
import matplotlib.pyplot as plt

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
random.seed(SEED)

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
HISTORY = []

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
    input = []
    # We input the distance to the target
    input.append(tl.distance_to_target(np.asarray(data.qpos[0:3]), np.asarray(TARGET_POSITION)))
    # We input the direction to the target
    input.extend(get_direction_vector(data))
    # We input the hinge angles
    input.extend(data.qpos[7:])
    # We input the hinge veloities
    input.extend(data.qvel[6:])
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

# The fitness functions that we test in our experimetn
FITNESS_FUNCTIONS = [
    "locomotion",
    "direct",
    "speed",
    "efficiency"]

PARAMETERS = ["w_" + f for f in FITNESS_FUNCTIONS]
COLUMNS = FITNESS_FUNCTIONS + PARAMETERS

# These are relevent for our threshold adaptation
STAGES = [
    {"locomotion": 1},
    {"locomotion": .5, "direct": .5},
    {"direct": 1},
    {"direct": .5, "speed": .5},
    {"speed": 1},
    {"speed": .5, "efficiency": .5},
    {"efficiency": 1},
]
LOOKBACK = 10
RATE_THRESHOLD = 0.1
STAGE = 0
BEST_IN_STAGE: list[float] = []

AMOUNT_OF_FUNCTIONS = len(FITNESS_FUNCTIONS)

# Gives
FITNESS_VALUES = {}
def fitness_function( #is there a way to make this more efficient?
    initial_position: npt.NDArray[np.float64],
    final_position: npt.NDArray[np.float64],
    total_control_effort: float,
    min_z_position: float,
    total_path_length: float,
    time_to_target: float | None,
    duration: float,
    min_distance_to_target: float,
    function
) -> float:
    """
    Determines the fitness value of the individual fitness functions
    (Easier to call and understand the code)
    :param initial_position:
    :param final_position:
    :param min_z_position:
    :param function:
    :return:
    """
    target = np.asarray(TARGET_POSITION)
    fitness = None
    ### changed fitness functions to the four we wanted, added necessary params to def and in the implementation below
    # we can derive the other data necisary
    if function == "efficiency":
        fitness = tl.fitness_distance_and_efficiency(initial_position, final_position, target, total_control_effort)
    elif function == "locomotion":
        fitness = tl.fitness_survival_and_locomotion(initial_position, final_position, target, min_z_position)
    elif function == "direct":
        fitness = tl.fitness_direct_path(initial_position, final_position, target, total_path_length)
    elif function == "speed":
        fitness = tl.fitness_speed_to_target(time_to_target,duration, min_distance_to_target)
    return fitness

def fitness(individual: Individual, method: str) -> float:
    # We take the sum off all these values
    fitness_value = individual.tags["locomotion"] * individual.tags["w_locomotion"]
    fitness_value += individual.tags["efficiency"] * individual.tags["w_efficiency"]
    fitness_value += individual.tags["direct"] * individual.tags["w_direct"]
    fitness_value += individual.tags["speed"] * individual.tags["w_speed"]
    return fitness_value

"""
Evaluation
"""

def evaluate_individual(weights: np.ndarray):
    """
    Determines the fitness of an individual.
    :param weights:
    :return: weights, fitness_dictionary
    """
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
        if len(weights) == 2:
            actions = nn_controller(m, d, weights)
        else:
            w1,w2, a = weights
            actions = nn_controller(m, d, (w1,w2))
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
    # Fitness measurements
    total_control_effort = 0.0
    total_path_length = 0.0
    min_z_position = np.inf
    min_distance_to_target = np.inf
    time_to_target = None

    target = np.asarray(TARGET_POSITION)

    # How close the robot needs to be to count as reaching the target.
    # do we have instructions on hwo to define this?
    TARGET_THRESHOLD = 0.1

    previous_position = get_core_position(data)

    for step in range(sim_steps):
        mj.mj_step(model, data)

        current_position = get_core_position(data)

        # ---------------------------------------------------------
        # Height
        # ---------------------------------------------------------
        min_z_position = min(min_z_position,current_position[2])

        # ---------------------------------------------------------
        # Total path length
        # Only count horizontal (x,y) movement.
        # ---------------------------------------------------------
        step_distance = np.linalg.norm(current_position[:2] - previous_position[:2])
        total_path_length += step_distance
        previous_position = current_position

        # ---------------------------------------------------------
        # Control effort
        # ---------------------------------------------------------
        total_control_effort += np.sum(data.ctrl ** 2)

        # ---------------------------------------------------------
        # Distance to target
        # Only x,y, matching the fitness functions.
        # ---------------------------------------------------------
        distance_to_target = np.linalg.norm(
            current_position[:2] - target[:2]
        )

        min_distance_to_target = min(min_distance_to_target,distance_to_target)

        # ---------------------------------------------------------
        # Time to target
        # ---------------------------------------------------------
        if (
            time_to_target is None
            and distance_to_target <= TARGET_THRESHOLD
        ):
            time_to_target = ((step + 1) * model.opt.timestep)

    duration = sim_steps * model.opt.timestep
    # --- Score -------------------------------------------------------------- #
    final_position = get_core_position(data)
    # --- Score -------------------------------------------------------------- #
    mj.set_mjcb_control(None)
    # We calculate the fitness w.r.t. all the fitness functions
    fitness_dictionary = {}
    for func in FITNESS_FUNCTIONS:
        fitness_dictionary[func] = fitness_function(initial_position, final_position, total_control_effort,
                                                    min_z_position, total_path_length, time_to_target, duration,
                                                    min_distance_to_target, func)
    return weights, fitness_dictionary
"""
Initialization
"""

def initialize_population(population_size:int=100,method:str = "locomotion") -> Population:
    # We randomly intitialize n bodies
    population = []
    for i in range(population_size):
        # For this genome, we create a individual
        ind = Individual()
        # We randomly create weights and evaluate their fitness
        weights, fitness_dictionary = evaluate_individual(None)
        # We add w_locomotion, w_efficiency, w_direct, w_speed to our dictionary (and to weights in case
        # of self adaptation)
        weights, tag = assign_parameter_weights(weights, fitness_dictionary, method)
        # We assign these to the individual
        ind.genotype = weights
        # We assign an index to the individuals for eay representation
        tag["index"] = i
        ind.tags = tag
        ind.fitness = fitness(ind, method)
        # We add the individual to the population
        population.append(ind)
    # We then create a population object and return it
    return Population(population)

def assign_parameter_weights(weights, tag, method):
    """"
    Adds w_locomotion, w_efficiency, w_direct, w_speed to the fitness (and in case of the
    self adaptation) to the weights, depending on which method we use
    """
    global PARAMETERS
    values = np.zeros(len(PARAMETERS))
    if method == "locomotion":
        index = FITNESS_FUNCTIONS.index("locomotion")
        values[index] = 1
    elif method == "efficiency":
        index = FITNESS_FUNCTIONS.index("efficiency")
        values[index] = 1
    elif method == "direct":
        index = FITNESS_FUNCTIONS.index("direct")
        values[index] = 1
    elif method == "speed":
        index = FITNESS_FUNCTIONS.index("speed")
        values[index] = 1
    elif method == "sum":
        # We want the sum to be normalized (more consistent)
        values = [0.25, 0.25, 0.25, 0.25]
    elif method == "threshold":
        values = values_for_threshold()
    elif method == "self-adaptive":
        if len(weights) == 2:
            # At the initialisation, the values are
            # uniform
            values = [0.25, 0.25, 0.25, 0.25]
            w1, w2 = weights
            weights = (w1,w2, values)
        else:
            # Otherwise we take the values
            w1,w2,values = weights
            val = np.array(values)
            # First we make sure that all values are positive
            min_val = np.min(val)
            if min_val < 0:
                val = val + abs(min_val)
            # We normalize the values
            normalized = (val - val.min()) / (val.max() - val.min())
            values = list(normalized)
    # We then add the parameter weights to the tag of the
    # individual
    for i, p in enumerate(PARAMETERS):
        tag[p] = values[i]
    return weights, tag

"""

"""

"""
Threshold adaptation
"""

def reset_threshold_state():
    """Call at the start of every run"""
    global STAGE, BEST_IN_STAGE
    STAGE = 0
    BEST_IN_STAGE = []

def values_for_threshold() -> list[float]:
    """Weights of the current stage, in the same order as PARAMETERS."""
    return [float(STAGES[STAGE].get(f, 0.0)) for f in FITNESS_FUNCTIONS]


def update_threshold_stage(pop) -> bool:
    """
    Records the best fitness, and if progress has stalled, moves to the next
    stage and rescores the whole population (no re-simulation needed, the raw
    values are in the tags). Returns True if the stage advanced.
    """
    global STAGE, BEST_IN_STAGE
    BEST_IN_STAGE.append(pop.best(sort="min", n=1)[0].fitness)

    if STAGE > len(STAGES)  or len(BEST_IN_STAGE) <= LOOKBACK:
        return False

    recent = BEST_IN_STAGE[-(LOOKBACK + 1):]
    rate = np.mean(np.diff(recent))
    if rate <= -RATE_THRESHOLD:
        return False

    STAGE += 1
    values = values_for_threshold()
    for ind in pop:
        for p, v in zip(PARAMETERS, values):
            ind.tags[p] = v
        ind.fitness = fitness(ind, "threshold")
    BEST_IN_STAGE = [min(ind.fitness for ind in pop)]
    return True

"""
Evolutionary steps
"""

def crossover(parents):
    """
    Simple one point crossover for two parents
    """
    p1, p2 = parents
    c1, c2 = [], []
    for a, b in zip(p1, p2):
        cut = int(RNG.integers(1, len(a)))
        c1.append(np.concatenate([b[:cut], a[cut:]]))
        c2.append(np.concatenate([a[:cut], b[cut:]]))
    return [c1, c2]

def mutate(weights, sigma: float = 0.1, rate: float = 1):
    """Gaussian mutation: each weight is perturbed with probability `rate`.

    Returns a new list of arrays; the input is left untouched.
    """
    mutated = []
    for w in weights:
        mask = RNG.random(w.shape) < rate
        noise = RNG.normal(0.0, sigma, size=w.shape)
        mutated.append(w + mask * noise)
    return mutated


def reproduction(population: Population, n_children=2, tournament_size:int = 4, parent_amount:int =2, method:str = "locomotion") -> Population:
    # We keep check of our children
    children = []
    # We perform tournament selection this amount of times
    for _ in range(n_children):
        parents = []
        # We perform n tournements
        for _ in range(parent_amount):
            # We take some random sample from the population
            tournament = population.sample(tournament_size)
            # We find the fittest in the population
            parents.extend(tournament.best(sort = "min", n=1))
        # We set the weights and fitness for all parents in an array
        weights = [p.genotype for p in parents]
        # We perform crossover to the parents
        weights_children = crossover(weights)
        # We create individuals using these weights and fitnesses
        for i,w in enumerate(weights_children):
            individual = Individual()
            mutated_weight = mutate(w)
            # We evaluate the new values of the fitness
            w_out, fitness_dict = evaluate_individual(mutated_weight)
            # We update the parameters
            # We give one parent for the copy of parameter values
            new_weights, tag = assign_parameter_weights(w_out,fitness_dict, method)
            individual.genotype = new_weights
            individual.tags = tag
            individual.fitness = fitness(individual, method)
            # We add the children to the population
            population.append(individual)
            children.append(individual)

    # We then kill of the individuals with the worst fitness
    worst = population.best(sort = "max", n=int(n_children*parent_amount))
    for ind in worst:
        ind.alive = False
    # We need to find the slots for our matrix creation (children have no index yet)
    free_slots = [ind.tags["index"] for ind in worst if "index" in ind.tags]
    # Surviving children take over those slots
    for child in children:
        if child.alive:
            child.tags["index"] = free_slots.pop()

    print(population.best(sort = "min", n=1)[0].fitness)
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
        w1, w2, a = weights
        actions = nn_controller(m, d, (w1,w2))

        # DIRECT application (see the controller contract above).
        d.ctrl[:] = actions

        # DELTA application - comment out the line above and use these instead:
        # delta = 0.05
        # d.ctrl[:] += actions * delta
        # d.ctrl[:] = np.clip(d.ctrl, -np.pi / 2, np.pi / 2)

    os.makedirs(DATA/"__videos__", exist_ok=True)
    mj.set_mjcb_control(control_callback)
    recorder = VideoRecorder(file_name=filename, output_folder=str(DATA / "__videos__"))
    video_renderer(
        model,
        data,
        duration=SIM_DURATION,
        video_recorder=recorder,
    )
    mj.set_mjcb_control(None)

"""
Experimental setup
"""

METHODS= ["locomotion",
           "speed",
           "efficiency",
           "direct",
           "sum",
           "threshold",
           "self-adaptive"]

def form_matrix(population:Population) -> np.array:
    """
    Turns the fitness values of the population into a matrix
    """
    # We go over all the individuals in the population
    # represent their fitness dictionary as a colmn of a matrix
    matrix = np.zeros((len(population), len(COLUMNS)))
    for ind in population:
        matrix[ind.tags["index"], :] = [ind.tags[c] for c in COLUMNS]
    return matrix


def individual_experimental(pop_size:int, time:int, seed:int, method:str="locomotion"):
    """
    This is where we call each individual experiment
    """
    global HISTORY
    reset_threshold_state()
    os.makedirs(DATA/"__history__", exist_ok=True)
    HISTORY = []
    pop = initialize_population(population_size=pop_size, method=method)
    best = pop.best(sort="min", n=1)[0]
    show_behavior(best, "start")
    HISTORY = [form_matrix(pop)]
    for i in range(time):
        if i % 10 == 0: print(".", end="")
        pop = reproduction(pop, method=method)
        if method == "threshold":
            update_threshold_stage(pop)
        HISTORY.append(form_matrix(pop))
    HISTORY = np.stack(HISTORY)
    np.save(DATA / "__history__" / f"{method}_seed{seed}.npy", HISTORY)
    best = pop.best(sort="min", n=1)[0]
    show_behavior(best, "finish")

def statistics(method:str, seed: int):
    #ToDo make the statistic statisizing
    # history has the form  (generations, pop_size, n_fitness_functions)
    history = np.load(DATA / "__history__" / f"{method}_seed{seed}.npy")
    # history: (generations, pop_size, len(COLUMNS))

    n = len(FITNESS_FUNCTIONS)
    time = np.arange(history.shape[0])

    # Mean over individuals -> one value per generation
    mean_fitness = {f: history[:, :, i].mean(axis=1) for i, f in enumerate(FITNESS_FUNCTIONS)}
    mean_weights = {f: history[:, :, n + i].mean(axis=1) for i, f in enumerate(FITNESS_FUNCTIONS)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for f in FITNESS_FUNCTIONS:
        ax1.plot(time, mean_weights[f], label=f"w_{f}")
    ax1.set_title(f"Mean parameter weights ({method}, seed {seed})")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Mean weight")
    ax1.legend()

    for f in FITNESS_FUNCTIONS:
        ax2.plot(time, mean_fitness[f], label=f)
    ax2.set_title(f"Mean raw fitness values ({method}, seed {seed})")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Mean fitness")
    ax2.legend()

    plt.tight_layout()
    os.makedirs(DATA / "__plots__", exist_ok=True)
    path = DATA / "__plots__" / f"{method}_seed{seed}.png"
    fig.savefig(path, dpi=150)
    print(f"Saved plot to {path}")
    plt.close(fig)
    return time, mean_fitness, mean_weights


def experimental_run(pop_size, time, amount_of_runs, initial_seed = 42):
    """
    This is where we run our experiment
    """
    global RNG
    # We define the different types of methods that we
    # want to test in our experiment:
    for method in METHODS:
        print("="*50)
        print(f"Running for method: {method}")
        seed = initial_seed
        for i in range(amount_of_runs):
            RNG = np.random.default_rng(seed)
            random.seed(seed)
            set_seed(seed)
            print(f"Experiment {i}:", end = " ")
            individual_experimental(pop_size, time, seed, method=method)
            seed += 1
            print("complete")
    print("="*50)
    print("Experiment complete")


if __name__ == "__main__":
    #individual_experimental(10, 20, seed = 42, method ="self-adaptive")
    history = np.load(DATA / "__history__" / f"self-adaptive_seed{42}.npy")
    statistics("self-adaptive", 42)
    # experimental_run(5, 5, amount_of_runs=2, initial_seed=42)
    # for method in METHODS:
    #     history = np.load(DATA / "__history__" / f"{method}_seed{42}.npy")
    #     print(history.shape)
    #     print(history[0][0])

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


# def fitness_per_method(individual: Individual, method:str)->float:
#     """
#     :param method: determines which method we use:
#     The individual functions:
#     locomotion, efficiency, direct, speed
#     The combined functions:
#     sum, threshold, self-adaptive
#     :return:
#     """
#     if method == "locomotion":
#         return individual.tags["locomotion"]
#     elif method == "efficiency":
#         return individual.tags["efficiency"]
#     elif method == "direct":
#         return individual.tags["direct"]
#     elif method == "speed":
#         return individual.tags["speed"]
#     elif method == "sum":
#         # we determine the values
#         return 10
#     elif method == "threshold":
#
#         a = individual.tags["locomotion"] * individual.tags["alpha"]
#         b = individual.tags["efficiency"] * individual.tags["beta"]
#         c = individual.tags["direct"] * individual.tags["gamma"]
#         d = individual.tags["speed"] * individual.tags["sigma"]
#         return a + b + c + d
#     elif method == "self-adaptive":
#         #ToDo create the self adaptive function
#         return 10
#     raise ValueError(f"Unknown method: {method}")


# def best_fitness_rate_of_change(lookback_size): #we look at the last n = 10 generations, if it doesnt chnage tahts a plateau
#     #we can aslo use this function to end the simulation if it stays the same for a long time
#     """
#     Calculates the mean rate of change of the best combined
#     fitness over the last n generations.
#
#     The combined fitness is the sum of the four fitness values.
#     Lower is better because this is a minimization problem.
#
#     Negative = improvement.
#     Positive = deterioration.
#     """
#     global HISTORY
#     if  len(HISTORY) < lookback_size:
#         return 100
#     # We take the last individuals
#     last = np.stack(HISTORY[-(lookback_size + 1):])
#     # We then calculate the fitness value for the
#     # given method. NOTE:  the last four values in the history
#     # correspond to the parameter weights
#     A = AMOUNT_OF_FUNCTIONS
#     fitness_per_individual = (last[:, :, :A] * last[:, :, A:2 * A]).sum(axis=2)
#     best_per_generation = fitness_per_individual.min(axis=1)
#     # Change between consecutive generations
#     changes = np.diff(best_per_generation)
#
#     # Mean rate of change
#     return changes.mean()

