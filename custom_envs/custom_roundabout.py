from __future__ import annotations

import numpy as np
from typing import Tuple, Dict, Any
from gymnasium.core import ObsType, ActType

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import CircularLane, LineType, SineLane, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.objects import RoadObject

class Pedestrian(RoadObject):
    LENGTH = 1
    WIDTH = 1

    def __init__(self, road, position, heading, speed=1.4):
        super().__init__(road, position, heading, speed)

    def act(self, action=None):
        pass

    def step(self, dt):
        vx = self.speed * np.cos(self.heading)
        vy = self.speed * np.sin(self.heading)
        self.position[0] += vx * dt
        self.position[1] += vy * dt

class AggressiveCar(IDMVehicle):
    LENGTH = 4
    WIDTH = 2
    MAX_SPEED = 30.0
    MAX_ACCELERATION = 4.0
    MIN_ACCELERATION = -8.0

class Truck(IDMVehicle):
    LENGTH = 7.0
    WIDTH = 3
    MAX_SPEED = 20.0
    MAX_ACCELERATION = 2.0
    MIN_ACCELERATION = -6.0

class Motorcycle(IDMVehicle):
    LENGTH = 2.2
    WIDTH = 1.5
    MAX_SPEED = 35.0
    MAX_ACCELERATION = 6.0
    MIN_ACCELERATION = -10.0



class CustomRoundaboutEnv(AbstractEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    "features_range": {
                        "x": [-100, 100],
                        "y": [-100, 100],
                        "vx": [-15, 15],
                        "vy": [-15, 15],
                    },
                },
                "action": {"type": "DiscreteMetaAction", "target_speeds": [0, 5, 10, 15, 20]},
                "incoming_vehicle_destination": None,
                "collision_reward": -3,
                "high_speed_reward": 0.2,
                "progress_reward": 0.1,
                "pedestrian_proximity_reward": -0.05,
                "right_lane_reward": 0,
                "lane_change_reward": -0.05,
                "screen_width": 600,
                "screen_height": 600,
                "centering_position": [0.5, 0.6],
                "duration": 15,
                "normalize_reward": False,
            }
        )
        return config

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step of the environment's dynamics, including vehicle simulation
        and dynamic pedestrian spawning.
        """
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Increment step count
        self.step_count += 1
        
        # Attempt to spawn a pedestrian at random times, only after the 3rd step
        if self.step_count >= 3:
            self._maybe_spawn_pedestrian()
        
        return obs, reward, terminated, truncated, info
    
    def _reward(self, action: int) -> float:
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * r for name, r in rewards.items()
        )
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [self.config["collision_reward"], self.config["high_speed_reward"]],
                [0, 1],
            )
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: int) -> dict[str, float]:
        
        current_lane = self.road.network.get_lane(self.vehicle.lane_index)
        
        longitudinal, _ = current_lane.local_coordinates(self.vehicle.position)
        
   
        
        progress_reward_value = longitudinal / 100.0 

        pedestrian_near_penalty = 0
        PEDESTRIAN_SAFE_DISTANCE = 5.0
        
        for v in self.road.vehicles:
            if isinstance(v, Pedestrian):
                distance = np.linalg.norm(self.vehicle.position - v.position)
                
                if distance < PEDESTRIAN_SAFE_DISTANCE:
                    pedestrian_near_penalty += (1 - distance / PEDESTRIAN_SAFE_DISTANCE)
                    
        return {
            "collision_reward": self.vehicle.crashed,
            "high_speed_reward": MDPVehicle.get_speed_index(self.vehicle)
            / (MDPVehicle.DEFAULT_TARGET_SPEEDS.size - 1),
            "progress_reward": progress_reward_value,
            "pedestrian_proximity_reward": -pedestrian_near_penalty, 
            "lane_change_reward": action in [0, 2],
            "on_road_reward": self.vehicle.on_road,
        }

    def _is_terminated(self) -> bool:
        return self.vehicle.crashed

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def _reset(self) -> None:
        self.step_count = 0
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        center = [0, 0]
        radius = 20
        alpha = 24

        net = RoadNetwork()
        radii = [radius, radius + 4]
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        line = [[c, s], [n, c]]

        for lane in [0, 1]:
            net.add_lane("se", "ex", CircularLane(center, radii[lane], np.deg2rad(90-alpha), np.deg2rad(alpha), clockwise=False, line_types=line[lane]))
            net.add_lane("ex", "ee", CircularLane(center, radii[lane], np.deg2rad(alpha), np.deg2rad(-alpha), clockwise=False, line_types=line[lane]))
            net.add_lane("ee", "nx", CircularLane(center, radii[lane], np.deg2rad(-alpha), np.deg2rad(-90+alpha), clockwise=False, line_types=line[lane]))
            net.add_lane("nx", "ne", CircularLane(center, radii[lane], np.deg2rad(-90+alpha), np.deg2rad(-90-alpha), clockwise=False, line_types=line[lane]))
            net.add_lane("ne", "wx", CircularLane(center, radii[lane], np.deg2rad(-90-alpha), np.deg2rad(-180+alpha), clockwise=False, line_types=line[lane]))
            net.add_lane("wx", "we", CircularLane(center, radii[lane], np.deg2rad(-180+alpha), np.deg2rad(-180-alpha), clockwise=False, line_types=line[lane]))
            net.add_lane("we", "sx", CircularLane(center, radii[lane], np.deg2rad(180-alpha), np.deg2rad(90+alpha), clockwise=False, line_types=line[lane]))
            net.add_lane("sx", "se", CircularLane(center, radii[lane], np.deg2rad(90+alpha), np.deg2rad(90-alpha), clockwise=False, line_types=line[lane]))

        # Access lanes: (r)oad/(s)ine
        access = 170
        dev = 85
        a = 5
        delta_st = 0.2*dev
        delta_en = dev - delta_st
        w = 2*np.pi/dev

        net.add_lane("ser", "ses", StraightLane([2,access],[2,dev/2],line_types=(s,c)))
        net.add_lane("ses","se", SineLane([2+a,dev/2],[2+a,dev/2-delta_st],a,w,-np.pi/2,line_types=(c,c)))
        net.add_lane("sx","sxs",SineLane([-2 - a, -dev / 2 + delta_en],[-2 - a, dev / 2],a,w,-np.pi / 2 + w * delta_en,line_types=(c, c)))
        net.add_lane("sxs", "sxr", StraightLane([-2, dev / 2], [-2, access], line_types=(n, c)))
        net.add_lane("eer", "ees", StraightLane([access, -2], [dev / 2, -2], line_types=(s, c)))
        net.add_lane("ees","ee",SineLane([dev / 2, -2 - a],[dev / 2 - delta_st, -2 - a],a,w,-np.pi / 2,line_types=(c, c)))
        net.add_lane("ex","exs",SineLane([-dev / 2 + delta_en, 2 + a],[dev / 2, 2 + a],a,w,-np.pi / 2 + w * delta_en,line_types=(c, c)))
        net.add_lane("exs", "exr", StraightLane([dev / 2, 2], [access, 2], line_types=(n, c)))
        net.add_lane("ner", "nes", StraightLane([-2, -access], [-2, -dev / 2], line_types=(s, c)))
        net.add_lane("nes","ne",SineLane([-2 - a, -dev / 2],[-2 - a, -dev / 2 + delta_st],a,w,-np.pi / 2,line_types=(c, c)))
        net.add_lane("nx","nxs",SineLane([2 + a, dev / 2 - delta_en],[2 + a, -dev / 2],a,w,-np.pi / 2 + w * delta_en,line_types=(c, c)))
        net.add_lane("nxs", "nxr", StraightLane([2, -dev / 2], [2, -access], line_types=(n, c)))
        net.add_lane("wer", "wes", StraightLane([-access, 2], [-dev / 2, 2], line_types=(s, c)))
        net.add_lane("wes","we",SineLane([-dev / 2, 2 + a],[-dev / 2 + delta_st, 2 + a],a,w,-np.pi / 2,line_types=(c, c)))
        net.add_lane("wx","wxs",SineLane([dev / 2 - delta_en, -2 - a],[-dev / 2, -2 - a],a,w,-np.pi / 2 + w * delta_en,line_types=(c, c)))
        net.add_lane("wxs", "wxr", StraightLane([-dev / 2, -2], [-access, -2], line_types=(n, c)))

        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def _make_vehicles(self) -> None:
        # speed_deviation = 1.0

        # Ego-vehicle
        ego_lane = self.road.network.get_lane(("ser","ses",0))
        ego_vehicle = self.action_type.vehicle_class(self.road, ego_lane.position(125.0,0.0), speed=8.0, heading=ego_lane.heading_at(140.0))
        try:
            ego_vehicle.plan_route_to("wxs") 
        except AttributeError:
            pass
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        # Other Vehicles: Spawn traffic directly onto the outer circular lane (lane 1)
        
        roundabout_lanes = [
            ("ex", "ee", 1), 
            ("nx", "ne", 1), 
            ("wx", "we", 1), 
            ("sx", "se", 1), 
        ]

        vehicle_pool = [AggressiveCar, Truck, Motorcycle]
        
        # buffer to keep vehicles off the immediate entry/exit transition points.
        BUFFER = 4.0 

        # Iterate over the roundabout segments to populate the environment
        for lane_index in roundabout_lanes:
            lane = self.road.network.get_lane(lane_index)
            
            VehicleClass = self.np_random.choice(vehicle_pool)

            positions = []
            
            if VehicleClass is Truck:
                # Trucks are limited to 1 and are spawned in the center of their segment.
                positions.append(lane.length / 2.0)
            else:
                # Spawn 1 or 2 cars/motorcycles. 2 with 75% probability
                num_vehicles = self.np_random.choice([1, 2], p=[0.25, 0.75])
                
                # Calculate the usable section of the lane accounting for the buffer
                usable_length = lane.length - 2 * BUFFER
                
                # Minimum separation required for two vehicles so they dont spawn on top of each other
                MIN_SAFE_SEPARATION = 8.0 
                
                if num_vehicles == 1 or usable_length < MIN_SAFE_SEPARATION:
                    # Single vehicle will be centered
                    positions.append(lane.length / 2.0)
                elif num_vehicles == 2:
                    # Two vehicles: placed near start/end of the usable area, using relative placement for guaranteed spacing.
                    
                    # Position 1 = BUFFER + (25% of usable length)
                    pos_start = BUFFER + usable_length * 0.25
                    # Position 2 = BUFFER + (75% of usable length)
                    pos_end = BUFFER + usable_length * 0.75
                    
                    positions.append(pos_start)
                    positions.append(pos_end)
                        
            # Spawn vehicles
            for longitudinal in positions:
                vehicle = VehicleClass.make_on_lane(
                    self.road,
                    lane_index,
                    longitudinal=longitudinal,
                    # speed=15.0 + self.np_random.normal() * speed_deviation
                    speed=15.0 # constant speed to avoid collisions
                )

                #Choose random exit for vehicles
                # destinations = ["wxr", "exr", "sxr", "nxr"]

                #Other vehicles do not exit at the same exit as ego vehicle
                #This is done to avoid other vehicles from crashing into the pedestrian
                destinations = ["exr", "sxr", "nxr"]

                destination = self.np_random.choice(destinations)
                vehicle.plan_route_to(destination)


                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)


    def _maybe_spawn_pedestrian(self) -> None:
        
        # Probability of spawning a pedestrian on any given step - 25% chance
        SPAWN_PROBABILITY = 0.25
        
        # List of all crossing lanes (between approach/exit roads and the roundabout itself)
        crossing_lanes = [
            # ("ses", "se", 0), 
            # ("sx", "sxs", 0), 
            # ("ees", "ee", 0), 
            # ("ex", "exs", 0), 
            # ("nes", "ne", 0), 
            # ("nx", "nxs", 0), 
            # ("wes", "we", 0), 
            ("wx", "wxs", 0)  #Only spawns in the exit that the ego vehicle goes to, to avoid other veghicles crashing
        ]
        
        if self.np_random.uniform() < SPAWN_PROBABILITY:
            # Randomly select a crossing lane
            random_index = self.np_random.choice(len(crossing_lanes))
            lane_idx = crossing_lanes[random_index]
            
            # Check if the spawn point is clear of existing pedestrians
            lane = self.road.network.get_lane(lane_idx)
            long = lane.length / 2
            # Calculate position at the crossing center
            pos = lane.position(long, -lane.width / 2) 
            
            is_clear = True
            for v in self.road.vehicles:
                if isinstance(v, Pedestrian):
                    distance = np.linalg.norm(pos - v.position)
                    if distance < 10.0:
                        is_clear = False
                        break
                        
            if is_clear:
                self.spawn_pedestrian_crossing(lane_idx)

    def spawn_pedestrian_crossing(self, lane_index):
        lane = self.road.network.get_lane(lane_index)

        long = lane.length / 2
        lateral_offset = -lane.width / 2
        pos = lane.position(long, lateral_offset)

        lane_heading = lane.heading_at(long)

        heading = lane_heading + np.pi/2

        ped = Pedestrian(
            road=self.road,
            position=pos,
            heading=heading,
            speed=0.3
        )

        self.road.vehicles.append(ped)