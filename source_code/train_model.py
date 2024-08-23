import argparse
import glob
import math
import os
import sys
import random
import time
import cv2
import numpy as np
from collections import deque
import tensorflow as tf
from threading import Thread

from tqdm import tqdm


# Searching the CARLA .egg file for the import
from models import cnn_4_layers_max_pooling, cnn_3x64_max_pooling, cnn_5_layers_max_pooling
from modified_tensorboard import ModifiedTensorBoard

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

sys.path.append(r'C:\Users\danie\Documents\DP_projekt\CARLA_0.9.11\WindowsNoEditor\PythonAPI\carla')
import carla
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.navigation.global_route_planner import GlobalRoutePlanner

tf.get_logger().setLevel('ERROR')

# Constants for the environment
SHOW_PREVIEW = False                        # True = opens a window displaying the models camera input
IM_WIDTH = 640                              # Agents camera window width
IM_HEIGHT = 480                             # Agents camera window height
TURN_OFF_RENDERING = False                  # UnrealEngine won't render the simulation graphics

# Constants for the training
EPISODES = 1000                             # Number of episodes that the model will train
SECONDS_PER_EPISODE = 30                    # Maximum duration of an episode in seconds
REPLAY_MEMORY_SIZE = 5000                   # Maximum number of samples that can be in the replay memory
MIN_REPLAY_MEMORY_SIZE = 1000               # Minimum number of samples in the replay memory to make a .fit()
MINIBATCH_SIZE = 16                         # Number of samples taken from the replay memory to update weights
PREDICTION_BATCH_SIZE = 1                   # Prediction batch size (DQN usually 1)
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 10                    # Number of episodes after the target model is updated
DISCOUNT = 0.99                             # Discount factor to determine the importance of the future rewards
epsilon = 1                                 # The epsilon value to ensure exploitation and exploration
EPSILON_DECAY = 0.9938                      # Decay of the epsilon after each episode (100-0.95, 200-0.975, 1000-0.9938)
MIN_EPSILON = 0.001                         # Value of the epsilon after which the agent is only exploiting

# Constants for logging
MODEL_NAME = "cnn_5_layers_max_pooling"       # Name of the model "Xception"
AGGREGATE_STATS_EVERY = 10                 # Number of episodes after we make a refresh the log stats
MIN_REWARD = 0                             # Minimum reward before we start logging

# Other constants
MEMORY_FRACTION = 0.4                       # Fraction of GPU memory used


def get_next_waypoint_angle_distance(vehicle_location, vehicle_rotation, next_waypoint):

    waypoint_location = next_waypoint.transform.location
    direction_vector = np.array([waypoint_location.x - vehicle_location.x, waypoint_location.y - vehicle_location.y])
    norm_direction_vector = direction_vector / np.linalg.norm(direction_vector)

    vehicle_forward_vector = np.array([np.cos(np.radians(vehicle_rotation)), np.sin(np.radians(vehicle_rotation))])
    dot_product = np.dot(norm_direction_vector, vehicle_forward_vector)
    angle = np.arccos(dot_product)  # Angle in radians

    distance = np.linalg.norm(direction_vector)
    angle = np.degrees(angle)  # Convert to degrees if preferred

    cross_product = np.cross(vehicle_forward_vector, norm_direction_vector)
    if cross_product < 0:
        angle = -angle

    return angle, distance


class CarlaEnvironment:

    STEER_AMT = 1.0     # Steering intensity

    def __init__(self, training_environment):
        # Initialize the connection to the CARLA simulator
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)

        # Set simulation map
        self.world = self.client.load_world('Town01')

        # Turn on or off simulator rendering
        settings = self.world.get_settings()
        settings.no_rendering_mode = TURN_OFF_RENDERING
        self.world.apply_settings(settings)
        self.training_environment = training_environment

        # Initialize blueprint library
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]

        # Initializing simulation data
        self.actor_list = []                # List to track the actors in the environment (eg. cars, sensors, ...)
        self.collision_hist = []            # List to track the actors collision events
        self.lane_invasion_hist = []        # List to track the actors lane invasions
        self.obstacle_detected = False      # Set to true if there is an obstacle ahead
        self.vehicle_speed = 0.0            # Initialize the vehicle speed

        # Set up route planning based on if we are training or evaluating
        self.map = self.world.get_map()
        self.spawn_points = self.world.get_map().get_spawn_points()
        self.roads = self.map.get_topology()

        # For the training we are spawning the agent at a random point on the map and set random destination
        if training_environment:
            start_point = random.choice(self.spawn_points)
            destination_point = random.choice(self.spawn_points)
        else:
            start_point = self.spawn_points[40]
            destination_point = self.spawn_points[50]

        # get the waypoints near to the spawnpoints
        self.start_waypoint = self.world.get_map().get_waypoint(start_point.location)
        self.end_waypoint = self.world.get_map().get_waypoint(destination_point.location)

        # Planning the path from the vehicle to the destination
        sampling_resolution = 8
        grpdao = GlobalRoutePlannerDAO(self.map, sampling_resolution)
        grp = GlobalRoutePlanner(grpdao)
        grp.setup()
        route = grp.trace_route(self.start_waypoint.transform.location, self.end_waypoint.transform.location)
        # Based on the path creating the waypoints leading to the destination
        self.waypoint_tracker = 1
        self.waypoint_list = [entry[0] for entry in route]


    def reset(self):
        # Reset function to start a new episode
        self.collision_hist = []        # Clear collision history
        self.actor_list = []            # Clear list of actors
        self.lane_invasion_hist = []    # Clear lane invasion history
        self.obstacle_detected = False  # Initialize obstacle detector
        self.vehicle_speed = 0.0        # Initialize the vehicle speed

        # Spawning the vehicle
        vehicle_not_spawned = True
        while vehicle_not_spawned:
            try:
                # Generating the spawn point for the agent vehicle
                if self.training_environment:
                    start_point = random.choice(self.spawn_points)
                    # start_point = self.spawn_points[14]
                else:
                    start_point = self.spawn_points[40]
                    end_point = self.spawn_points[50]
                self.vehicle = self.world.spawn_actor(self.model_3, start_point)
                vehicle_not_spawned = False
            except RuntimeError as e:
                print(f"Spawn failed: {e}")
        self.actor_list.append(self.vehicle)

        # Set up camera sensor
        # Possible camera sensors: rgb/depth/semantic_segmentation
        self.semantic_cam_bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        self.semantic_cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
        self.semantic_cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
        self.semantic_cam_bp.set_attribute("fov", f"110")
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.semantic_cam_sensor = self.world.spawn_actor(self.semantic_cam_bp, transform, attach_to=self.vehicle)
        self.actor_list.append(self.semantic_cam_sensor)
        self.semantic_cam_sensor.listen(lambda data: self.process_img(data))

        # Set up collision sensor and attach to vehicle
        collision_sensor_bp = self.blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(collision_sensor_bp, transform, attach_to=self.vehicle)
        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(lambda event: self.collision_data(event))

        # Set up lane invasion sensor
        lane_invasion_sensor_bp = self.blueprint_library.find("sensor.other.lane_invasion")
        lane_invasion_sensor_transform = carla.Transform(carla.Location(x=0.0, z=2.5))
        self.lane_invasion_sensor = self.world.spawn_actor(lane_invasion_sensor_bp, lane_invasion_sensor_transform,
                                                           attach_to=self.vehicle)
        self.actor_list.append(self.lane_invasion_sensor)
        self.lane_invasion_sensor.listen(lambda event: self.lane_invasion_data(event))

        # Set up obstacle detection sensor
        obstacle_sensor_bp = self.blueprint_library.find("sensor.other.obstacle")
        obstacle_sensor_bp.set_attribute("distance", "5")  # Max distance in meters to detect obstacles
        obstacle_sensor_bp.set_attribute("hit_radius", "0.5")  # Radius in meters to consider an obstacle hit
        obstacle_sensor_bp.set_attribute("sensor_tick", "0.3")  # Sensor update frequency in seconds
        obstacle_sensor_location = carla.Location(x=2.5, z=0.7)
        obstacle_sensor_transform = carla.Transform(obstacle_sensor_location)
        self.obstacle_sensor = self.world.spawn_actor(obstacle_sensor_bp, obstacle_sensor_transform, attach_to=self.vehicle)
        self.actor_list.append(self.obstacle_sensor)
        self.obstacle_sensor.listen(lambda event: self.obstacle_data(event))

        # Initialize vehicle controls
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        # Retrieving the map infrastructure
        self.waypoint_list = []
        self.map = self.world.get_map()
        self.spawn_points = self.world.get_map().get_spawn_points()
        self.roads = self.map.get_topology()
        self.start_waypoint = self.world.get_map().get_waypoint(start_point.location)
        sampling_resolution = 10
        grpdao = GlobalRoutePlannerDAO(self.map, sampling_resolution)
        grp = GlobalRoutePlanner(grpdao)
        grp.setup()
        # Generating the waypoints leading to destination
        while len(self.waypoint_list) < 2:
            if self.training_environment:
                random_spawn_point = random.choice(self.spawn_points)
                self.end_waypoint = self.world.get_map().get_waypoint(random_spawn_point.location)
                route = grp.trace_route(self.start_waypoint.transform.location, self.end_waypoint.transform.location)
                self.waypoint_list = [entry[0] for entry in route]
            else:
                self.end_waypoint = self.world.get_map().get_waypoint(end_point.location)
                route = grp.trace_route(self.start_waypoint.transform.location, self.end_waypoint.transform.location)
                self.waypoint_list = [entry[0] for entry in route]

        # For debug purposes drawing the way points
        for uwp in self.waypoint_list:
            self.world.debug.draw_string(uwp.transform.location, '^', draw_shadow=False,
                                         color=carla.Color(r=0, g=0, b=255), life_time=30.0,
                                         persistent_lines=True)

        # Setting the next waypoint that the agent needs to reach
        self.next_waypoint = self.waypoint_list[1]
        self.waypoint_tracker = 1
        # Calculating the distance and the angle between the waypoint and the vehicle
        self.vehicle_location = self.vehicle.get_location()
        self.vehicle_rotation = self.vehicle.get_transform().rotation.yaw
        angle, distance = get_next_waypoint_angle_distance(self.vehicle_location, self.vehicle_rotation, self.next_waypoint)
        self.previous_distance_to_waypoint = self.vehicle.get_location().distance(self.next_waypoint.transform.location)

        # Wait for everything to be set up properly
        time.sleep(3)

        # Waiting for the camera first camera image to be ready
        while self.front_camera is None:
            time.sleep(0.01)

        # Reset episode timer
        self.episode_start = time.time()

        return (self.front_camera, [angle, distance, self.vehicle_speed])

    def process_img(self, image):
        # Processing the image data from the camera sensor
        image.convert(carla.ColorConverter.CityScapesPalette)
        i = np.array(image.raw_data)                # Creating a numpy array from the sensor image data
        i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))    # Reshaping the array so it separates the RGBA channels and img size
        i3 = i2[:, :, :3]                           # Dropping the alpha channel
        if SHOW_PREVIEW:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3

    def obstacle_data(self, event):
        # Obstacle detection sensor for the reward policy
        if event.distance < 6:
            self.obstacle_detected = True
        else:
            self.obstacle_detected = False

    def collision_data(self, event):
        # Recording collisions
        self.collision_hist.append(event)

    def lane_invasion_data(self, event):
        # Recording lane invasions
        self.lane_invasion_hist.append(event)

    def update_waypoint(self):
        # Updates the vehicle location for the next step
        self.vehicle_location = self.vehicle.get_location()
        self.vehicle_rotation = self.vehicle.get_transform().rotation.yaw

    def step(self, step_action):
        # The agent interaction with the environment
        if step_action == 0:        # apply throttle
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.75, steer=0))
        elif step_action == 1:      # hard right turn
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=0.5 * self.STEER_AMT))
        elif step_action == 2:      # hard left turn
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=-0.5 * self.STEER_AMT))
        elif step_action == 3:      # apply brake
            self.vehicle.apply_control(carla.VehicleControl(brake=1.0))
        elif step_action == 4:  # no action
            pass

        step_reward = 0.0
        step_done = False

        self.update_waypoint()
        angle, distance = get_next_waypoint_angle_distance(self.vehicle_location, self.vehicle_rotation,
                                                           self.next_waypoint)
        # Giving reward if the vehicle got closer to the waypoint and penalty if got further
        # Also giving reward for reaching the waypoints
        if distance <= 3.0:
            if self.waypoint_tracker == len(self.waypoint_list) - 1:
                step_reward += 1
                step_done = True
                print("Last waypoint reached")
            else:
                step_reward += 1.0
                self.waypoint_tracker += 1
                self.next_waypoint = self.waypoint_list[self.waypoint_tracker]
                self.previous_distance_to_waypoint = self.vehicle.get_location().distance(self.next_waypoint.transform.location)
        elif distance - self.previous_distance_to_waypoint < 0:
            step_reward += 0.5
            self.previous_distance_to_waypoint = distance
        else:
            step_reward -= 0.5
            self.previous_distance_to_waypoint = distance

        # Drawing the next waypoint for debug and visual purposes
        self.world.debug.draw_string(self.next_waypoint.transform.location, '^', draw_shadow=False,
                                color=carla.Color(r=255, g=0, b=0), life_time=3.0,
                                persistent_lines=True)

        # Reward for being aligned with the next waypoint
        if abs(angle) < 10:
            step_reward += 0.5

        # Calculating speed from velocity
        v = self.vehicle.get_velocity()
        self.vehicle_speed = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))

        # Reward for maintaining the optimal speed
        target_speed = 30
        if target_speed - 10 <= self.vehicle_speed <= target_speed + 10:
            step_reward += 0.5

        if self.vehicle_speed > 50:
            step_reward -= 0.5

        # Encourage vehicle to don't brake if it's unnecessary
        if step_action == 3 or step_action == 4:
            if not self.obstacle_detected:
                step_reward -= 1.0
            else:
                step_reward += 0.5

        # Encourage the vehicle to stay between the lines and don't invade them
        if len(self.lane_invasion_hist) != 0:
            step_reward -= 0.5
            self.lane_invasion_hist.clear()

        # Big penalty for collisions
        if len(self.collision_hist) != 0:
            step_reward -= 5.0
            step_done = True

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            step_done = True

        return (self.front_camera, [angle, distance, self.vehicle_speed]), step_reward, step_done


class DQNAgent:
    def __init__(self):
        # Creating two models to stabilize the learning
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)       # Stores the experiences of the agent

        # TensorBoard for logging
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.graph = tf.get_default_graph()
        self.last_logged_episode = 0

        self.target_update_counter = 0                              # Tracks when to update the target model
        self.terminate = False                                      # Flag to terminate the training loop
        self.training_initialized = False                           # Flag to indicate if the training is initialized

    def create_model(self):
        # Returns the model based on the MODEL_NAME
        input_shape = (IM_HEIGHT, IM_WIDTH, 3)
        if MODEL_NAME == "cnn_4_layers_max_pooling":
            return cnn_4_layers_max_pooling()
        elif MODEL_NAME == "cnn_3x64_max_pooling":
            return cnn_3x64_max_pooling()
        elif MODEL_NAME == "cnn_5_layers_max_pooling":
            return cnn_5_layers_max_pooling()

    def update_replay_memory(self, transition):
        # Adds transition to the replay memory
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def train(self):
        # Train the model only when there are enough transitions in the replay memory
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states_images = np.array([transition[0][0] for transition in minibatch]) / 255
        current_states_info = np.array([transition[0][1] for transition in minibatch])
        current_states = [current_states_images, current_states_info]
        with self.graph.as_default():
            current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)

        new_current_states_images = np.array([transition[3][0] for transition in minibatch]) / 255
        new_current_states_info = np.array([transition[3][1] for transition in minibatch])
        new_current_states = [new_current_states_images, new_current_states_info]
        with self.graph.as_default():
            future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE)

        X = []  # States
        y = []  # Target Q-values

        # Update Q-values for each state based on the reward and the maximum future Q-value
        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step

        X_images = [state[0] for state in X]
        X_additional_info = [state[1] for state in X]
        # Fit the model to the updated Q-values
        with self.graph.as_default():
            self.model.fit([np.array(X_images), np.array(X_additional_info)], np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)

        if log_this_step:
            self.target_update_counter += 1

        # Updating target model if there were enough episodes
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        # Return the Q-values for a given stat using the main model
        image_data, additional_info = state[0], state[1]
        image_data = np.array(image_data.reshape(-1, *state[0].shape)/255)
        additional_info = np.array(additional_info).reshape(-1, 3)
        return self.model.predict([image_data, additional_info])[0]

    def train_in_loop(self):
        # Train the model in loops
        # Initialize the model because the first tends to take long
        X_image = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)
        X_additional_info = np.random.uniform(low=-1, high=1, size=(1, 3)).astype(np.float32)
        y = np.random.uniform(size=(1, 5)).astype(np.float32)
        with self.graph.as_default():
            self.model.fit([X_image, X_additional_info], y, verbose=False, batch_size=1)

        self.training_initialized = True
        # Training in loop while terminate flag is False
        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)


if __name__ == '__main__':

    FPS = 60
    # For stats
    ep_rewards = [0]
    episode_times_list = []

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    tf.set_random_seed(1)

    # Memory fraction, used mostly when training multiple agents
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    # backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    # Create models folder
    if not os.path.isdir('models'):
        os.makedirs('models')

    # Create agent and environment
    agent = DQNAgent()
    env = CarlaEnvironment(training_environment=True)

    # Start training thread and wait for training to be initialized
    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    while not agent.training_initialized:
        time.sleep(0.01)

    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        # Update tensorboard step every episode
        agent.tensorboard.step = episode
        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1
        # Reset environment and get initial state
        current_state = env.reset()
        # Clear the collision history, because spawning the vehicle might produce collision
        env.collision_hist = []
        # Reset flag and start iterating until episode ends
        done = False
        episode_start = time.time()
        # Play for given number of seconds only
        while True:
            # Choosing a prediction or a random action based on the epsilon
            if np.random.random() > epsilon:
                # Get action from Q table
                action = np.argmax(agent.get_qs(current_state))
            else:
                # Get random action
                action = np.random.randint(0, 5)
                time.sleep(1/FPS)
            # Executing the step based on the prediction or random
            new_state, reward, done = env.step(action)
            # Calculating the episode reward
            episode_reward += reward
            # Updating replay memory after step
            agent.update_replay_memory((current_state, action, reward, new_state, done))

            current_state = new_state
            step += 1
            if done:
                break
        episode_end = time.time()
        # Destroying agents at the end of episode
        for actor in env.actor_list:
            actor.destroy()

        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        episode_times_list.append(episode_end-episode_start)
        if not episode % AGGREGATE_STATS_EVERY:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            average_time = sum(episode_times_list[-AGGREGATE_STATS_EVERY:])/len(episode_times_list[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon, average_episode_time=average_time)

            # Save model, but only when min reward is greater or equal a set value
            if min_reward >= MIN_REWARD:
                agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        # Decay epsilon to maintain exploitation and exploration
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

    # Set termination flag for training thread and wait for it to finish
    agent.terminate = True
    trainer_thread.join()
    agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')


