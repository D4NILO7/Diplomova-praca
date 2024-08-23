import csv
import glob
from collections import deque
import numpy as np
import cv2
import time
from keras.models import load_model
from train_model import CarlaEnvironment


# MODEL_PATH = 'models/cnn_3x64_average_pooling___171.10max___83.63avg___34.80min__1712442300.model'
EVAL_MODELS_DIR = 'eval-konz/'
NUM_EVAL_EPISODES = 10
SHOW_CAM_PREVIEW = True
FILE_NAME = "evaluation_with_npc"


def get_evaluation_model_paths(directory_path, extension = '*.model'):

    search_pattern = directory_path + extension
    model_paths = glob.glob(search_pattern)
    return model_paths

if __name__ == '__main__':

    # Memory fraction
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    # backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    # Load the model paths
    model_paths = get_evaluation_model_paths(EVAL_MODELS_DIR)

    # Create environment
    env = CarlaEnvironment(training_environment=False)

    # For agent speed measurements - keeps last 60 frametimes
    fps_counter = deque(maxlen=60)

    models_wp_performance = {}
    model_avg_lane_invasion_performance = {}
    model_counter = 1
    with open(FILE_NAME, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model',
                         'avg_wp_reached',
                         'avg_lane_invasion',
                         'avg_step_reward',
                         'avg_total_reward',
                         'avg_speed_dest_reached',
                         'avg_time_destination_reached',
                         'num_of_collisions',
                         'destination_reached'])
        # Loop trough all the models
        for path in model_paths:
            # Load the next model
            model = load_model(path)
            print("-------------------------------------------------")
            print(f"[+] Starting the evaluation of model: {path}")

            # Init the evaluation data
            model_avg_rewards_per_step = []              # List of average step rewards per episode
            model_episode_times = []            # List of episode durations
            model_waypoints_reached = []        # List of waypoints reached during evaluation
            model_destination_reached = 0       # Number of times the model reached the destination
            model_avg_speed = 0                 # Models average speed
            model_avg_speed_dest_reached = []   # All the avg speeds when the model reached the destination
            model_percentage_of_wp_reached_list = []    # Percentage of waypoints reached during episodes
            model_collision_counter = 0
            model_lane_invasion_list = []
            model_avg_total_reward = []

            graph_name = f"model {model_counter}"
            model_counter += 1

            for episode in range(1, NUM_EVAL_EPISODES + 1):

                print(f"[+] Episode {episode}/{NUM_EVAL_EPISODES}")
                print("[+] Resetting the environment")

                # Reset environment and get initial state
                current_state = env.reset()
                done = False
                env.collision_hist = []
                total_rewards = 0
                step_counter = 0
                model_avg_speed = 0

                episode_start = time.time()
                while not done:
                    # For FPS counter
                    step_start = time.time()

                    # Retrieving current state data
                    image_data, additional_info = current_state[0], current_state[1]
                    additional_info = np.array(additional_info).reshape(-1, 3)

                    # Show agent camera view
                    if SHOW_CAM_PREVIEW:
                        cv2.imshow(f'Agent - preview', image_data)
                        cv2.waitKey(1)
                        image_data = np.array(image_data.reshape(-1, *current_state[0].shape) / 255)

                    # Predict an action based on current observation space
                    qs = model.predict([image_data, additional_info])[0]
                    action = np.argmax(qs)

                    # Step environment
                    new_state, reward, done = env.step(action)
                    # Set current step for next loop iteration
                    current_state = new_state

                    total_rewards += reward
                    step_counter += 1

                    model_avg_speed += env.vehicle_speed

                    # Measure step time, append to a deque, then print mean FPS for last 60 frames, q values and taken action
                    frame_time = time.time() - step_start
                    fps_counter.append(frame_time)
                    # print(f'Agent: {len(fps_counter)/sum(fps_counter):>4.1f} FPS | Action: [{qs[0]:>5.2f}, {qs[1]:>5.2f}, {qs[2]:>5.2f}, {qs[3]:>5.2f}] {action}')

                episode_end = time.time()

                # Append episode times, waypoints reached
                model_waypoints_reached.append(env.eval_waypoint_reached)
                model_percentage_of_wp_reached_list.append(env.eval_waypoint_reached / env.eval_num_of_waypoints * 100)
                model_lane_invasion_list.append(env.eval_lane_invasion_counter)
                model_avg_total_reward.append(total_rewards)
                if env.eval_crash_test:
                    model_collision_counter += 1
                # Calculate and append the episode average reward per step
                if step_counter != 0:
                    model_avg_rewards_per_step.append(total_rewards / step_counter)
                else:
                    model_avg_rewards_per_step.append(0.0)

                if env.eval_dest_reached:
                    model_destination_reached += 1
                    model_avg_speed_dest_reached.append(model_avg_speed/step_counter)
                    model_episode_times.append(episode_end - episode_start)

                models_wp_percentage = env.eval_waypoint_reached / env.eval_num_of_waypoints * 100
                # print(f"time: {model_episode_times}")
                # print(model_waypoints_reached)
                # print(f"average speed: {model_avg_speed/step_counter}")
                # print(model_collision_counter)
                # print(f"total reward {total_rewards}")
                # print(f"number of line invasions: {env.eval_lane_invasion_counter}")
                # print(f"average reward: {model_avg_rewards_per_step}")

                # Destroy an actor at end of episode
                for actor in env.actor_list:
                    actor.destroy()

            avg_lane_invasion = round(sum(model_lane_invasion_list)/len(model_lane_invasion_list), 2)
            avg_percentage_wp_reached = round(sum(model_percentage_of_wp_reached_list) / len(model_percentage_of_wp_reached_list), 2)
            avg_step_reward = round(sum(model_avg_rewards_per_step)/len(model_avg_rewards_per_step), 2)
            avg_total_reward = round(sum(model_avg_total_reward)/len(model_avg_total_reward), 2)

            if not len(model_avg_speed_dest_reached) == 0:
                avg_speed_dest_reached = round(sum(model_avg_speed_dest_reached)/len(model_avg_speed_dest_reached), 2)
            else:
                avg_speed_dest_reached = 0.0

            if not len(model_episode_times) == 0:
                avg_time_destination_reached = round(sum(model_episode_times)/len(model_episode_times), 2)
            else:
                avg_time_destination_reached = 0.0

            print("--------------------------------------------")
            print(f"model:{path}")
            print(f"avg_lane_invasion:{avg_lane_invasion},")
            print(f"avg_wp_reached:{avg_percentage_wp_reached}%,")
            print(f"avg_step_reward:{avg_step_reward},")
            print(f"avg_total_reward: {avg_total_reward},")
            print(f"num_of_collisions: {model_collision_counter},")
            print(f"destination_reached: {model_destination_reached}")
            print(f"avg_speed_dest_reached: {avg_speed_dest_reached}")
            print("---------------------------------------------")
            writer.writerow([path,
                             avg_percentage_wp_reached,
                             avg_lane_invasion,
                             avg_step_reward,
                             avg_total_reward,
                             avg_speed_dest_reached,
                             avg_time_destination_reached,
                             model_collision_counter,
                             model_destination_reached])


